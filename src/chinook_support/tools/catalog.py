"""Workflow 2 - browsing the catalog and getting recommendations.

`recommend_tracks` is built by a factory that closes over which version of the
recommendation contract to use. The tool's name, description, signature, and the
prompt around it are identical in both versions - only the SQL underneath differs.

That is deliberate. When the demo switches from v1 to v2, nothing the model can
see has changed. It rules out "you fixed it by rewording something."
"""

from langchain.tools import ToolRuntime, tool

from .. import queries
from ..context import Ctx
from ..contracts import ToolResult, ambiguous, no_match, not_available, ok


@tool
def search_catalog(
    search_term: str,
    genre: str | None,
    runtime: ToolRuntime[Ctx],
) -> ToolResult:
    """Search the store's catalog by song, artist, or album name.

    Args:
        search_term: What to look for - a song title, artist, or album.
        genre: Optional genre to narrow the search, e.g. "Jazz". Use null for all.
    """
    # Several different songs can share a title. Return the choices rather than
    # silently picking one - a guess here becomes a confident wrong answer later.
    same_name = queries.exact_title_artists(search_term)
    artists = {row["artist"] for row in same_name}
    if len(artists) > 1:
        return ambiguous(
            same_name,
            f'"{search_term}" is the name of {len(artists)} different songs in our '
            f"catalog. Ask which artist they meant before going further.",
            matched_exact_title=True,
        )

    rows = queries.search_tracks(search_term, genre, limit=10)
    if not rows:
        return no_match(
            f'Nothing in the catalog matches "{search_term}"'
            + (f" in {genre}." if genre else "."),
            next_actions=[
                "Tell the customer the store doesn't carry this.",
                "Do not invent a title. Offer to search for something else.",
            ],
            genre=genre,
        )

    return ok(rows, f'{len(rows)} catalog match(es) for "{search_term}".', genre=genre)


def make_recommend_tracks(*, exclude_owned: bool):
    """Build the recommendation tool.

    `exclude_owned=False` is the v1 contract: it ranks by store-wide sales and
    never joins back to this customer's purchases. `exclude_owned=True` is v2.
    """

    @tool
    def recommend_tracks(
        seed_invoice_id: int,
        genre: str | None,
        max_price: float | None,
        runtime: ToolRuntime[Ctx],
    ) -> ToolResult:
        """Recommend tracks the customer might like, based on one of their orders.

        Args:
            seed_invoice_id: An order of theirs to base suggestions on.
            genre: Optional genre filter, e.g. "Jazz". Use null for no filter.
            max_price: Optional maximum price per track. Use null for no limit.
        """
        customer_id = runtime.context.customer_id

        # Same ownership gate as get_my_invoice - you can only seed from your own
        # order. Note this check exists in BOTH versions: the v1 bug is about
        # recommendation quality, not about the isolation boundary.
        if queries.invoice_header(customer_id, seed_invoice_id) is None:
            return not_available(
                f"Order {seed_invoice_id} isn't on this account.",
                next_actions=["Offer to list the orders that are on this account."],
            )

        seed = queries.invoice_seed_profile(seed_invoice_id)
        rows = queries.recommend(
            customer_id,
            seed,
            genre=genre,
            max_price=max_price,
            limit=5,
            exclude_owned=exclude_owned,
        )

        if not rows:
            return no_match(
                "Nothing in the catalog matches those constraints.",
                next_actions=["Suggest relaxing the genre or price limit."],
                genre=genre,
                max_price=max_price,
                excluded_previously_purchased=exclude_owned,
            )

        return ok(
            rows,
            f"{len(rows)} suggestion(s) based on order {seed_invoice_id}.",
            # These are the guarantees the TOOL enforced. They end up in the trace
            # and are what the evaluators assert against - so "did the filter run?"
            # is a fact you can look up, not a thing you hope about.
            seed_invoice_id=seed_invoice_id,
            genre=genre,
            max_price=max_price,
            excluded_previously_purchased=exclude_owned,
            ranked_by="store_wide_sales",
        )

    return recommend_tracks
