import asyncpraw
import pandas as pd

from src.config import Settings


def create_reddit_client(settings: Settings) -> asyncpraw.Reddit:
    return asyncpraw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
    )


async def fetch_reddit_posts(
    reddit: asyncpraw.Reddit,
    subreddit_name: str,
    search_terms: list[str],
    limit: int = 100,
) -> pd.DataFrame:
    posts = []
    subreddit = await reddit.subreddit(subreddit_name)

    for term in search_terms:
        async for post in subreddit.search(
            term,
            sort="new",
            time_filter="month",
            limit=limit,
        ):
            posts.append(
                {
                    "post_id": post.id,
                    "search_term": term,
                    "title": post.title,
                    "body": post.selftext,
                    "num_comments": post.num_comments,
                    "created_at": pd.to_datetime(post.created_utc, unit="s"),
                    "reddit_url": f"https://reddit.com{post.permalink}",
                    "external_url": post.url,
                }
            )

    reddit_data = pd.DataFrame(posts)

    if not reddit_data.empty:
        reddit_data = reddit_data.drop_duplicates(subset="post_id").reset_index(drop=True)

    return reddit_data
