"""RSS news ingestion with VADER sentiment scoring."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from predictor.core.event_bus import EventBus

logger = logging.getLogger(__name__)

# Event type for news
NEWS_EVENT = "news.article"


class NewsArticle:
    """Parsed news article with sentiment score."""

    def __init__(
        self,
        title: str,
        url: str,
        summary: str,
        published: datetime | None,
        source: str,
        sentiment_compound: float,
        matched_tickers: list[str],
    ) -> None:
        self.title = title
        self.url = url
        self.summary = summary
        self.published = published
        self.source = source
        self.sentiment_compound = sentiment_compound
        self.matched_tickers = matched_tickers


# Keyword -> event series mapping for Kalshi markets
# Extend this as needed for the markets you trade
_KEYWORD_MAP: dict[str, list[str]] = {
    "federal reserve": ["FED", "FOMC"],
    "fed funds": ["FED", "FOMC"],
    "interest rate": ["FED", "FOMC"],
    "fomc": ["FED", "FOMC"],
    "inflation": ["CPI", "INFLATION"],
    "consumer price": ["CPI"],
    "jobs report": ["JOBS", "UNRATE"],
    "unemployment": ["UNRATE", "JOBS"],
    "nonfarm payroll": ["JOBS"],
    "gdp": ["GDP"],
    "gross domestic": ["GDP"],
    "recession": ["GDP", "RECESSION"],
    "election": ["PRES", "SENATE", "HOUSE"],
    "president": ["PRES"],
    "senate": ["SENATE"],
    "congress": ["HOUSE", "SENATE"],
    "trump": ["PRES", "TRUMP"],
    "biden": ["PRES", "BIDEN"],
    "stock market": ["SP500", "NASDAQ"],
    "s&p 500": ["SP500"],
    "nasdaq": ["NASDAQ"],
    "oil price": ["OIL", "WTI"],
    "crude oil": ["OIL", "WTI"],
    "bitcoin": ["BTC", "CRYPTO"],
    "cryptocurrency": ["CRYPTO", "BTC"],
}


class NewsFeedService:
    """Ingests news from RSS feeds, scores sentiment, and publishes events.

    Periodically polls configured RSS feeds, extracts articles,
    computes VADER sentiment scores, matches to relevant Kalshi
    market series, and publishes NewsArticle events.
    """

    def __init__(
        self,
        rss_feeds: list[str],
        event_bus: EventBus,
        poll_interval: int = 300,
    ) -> None:
        self._feeds = rss_feeds
        self._event_bus = event_bus
        self._poll_interval = poll_interval
        self._analyzer = SentimentIntensityAnalyzer()
        self._seen_urls: set[str] = set()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Begin polling RSS feeds."""
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "NewsFeedService started: %d feeds, %ds interval",
            len(self._feeds), self._poll_interval,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _poll_loop(self) -> None:
        """Periodically poll all feeds."""
        # Do an initial poll immediately
        await self._poll_all_feeds()
        while True:
            await asyncio.sleep(self._poll_interval)
            await self._poll_all_feeds()

    async def _poll_all_feeds(self) -> None:
        """Poll all configured RSS feeds."""
        for feed_url in self._feeds:
            try:
                await self._poll_feed(feed_url)
            except Exception:
                logger.exception("Error polling feed: %s", feed_url)

    async def _poll_feed(self, feed_url: str) -> None:
        """Parse a single RSS feed and process new articles."""
        # feedparser is synchronous; run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, feed_url)

        source = parsed.feed.get("title", feed_url)
        new_count = 0

        for entry in parsed.entries:
            url = entry.get("link", "")
            if not url or url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            published = self._parse_published(entry)

            # Compute sentiment
            text = f"{title}. {summary}"
            scores = self._analyzer.polarity_scores(text)
            compound = scores["compound"]

            # Match to Kalshi markets
            matched = self._match_to_markets(text)
            if not matched:
                continue  # Skip articles that don't match any tracked markets

            article = NewsArticle(
                title=title,
                url=url,
                summary=summary[:500],
                published=published,
                source=source,
                sentiment_compound=compound,
                matched_tickers=matched,
            )

            await self._event_bus.publish(NEWS_EVENT, article)
            new_count += 1

        if new_count > 0:
            logger.info("Feed %s: %d new articles matched", source, new_count)

    def _match_to_markets(self, text: str) -> list[str]:
        """Match article text to Kalshi market series using keyword map."""
        text_lower = text.lower()
        matched_series: set[str] = set()

        for keyword, series_list in _KEYWORD_MAP.items():
            if keyword in text_lower:
                matched_series.update(series_list)

        return list(matched_series)

    @staticmethod
    def _parse_published(entry: Any) -> datetime | None:
        """Parse the published date from a feed entry."""
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            try:
                from time import mktime
                return datetime.fromtimestamp(mktime(published_parsed), tz=UTC)
            except Exception:
                pass
        return None
