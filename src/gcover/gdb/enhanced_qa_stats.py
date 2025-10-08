#!/usr/bin/env python3
"""
Enhanced QA Statistics module with trend analysis and improved filtering.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import duckdb
import geopandas as gpd
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
from loguru import logger

QATestType = Literal["Topology", "TechnicalQualityAssurance"]
RCVersion = Literal["2016-12-31", "2030-12-31"]


@dataclass
class TrendData:
    """Represents trend data for a specific test."""

    current_count: int
    previous_count: Optional[int]
    current_date: datetime
    previous_date: Optional[datetime]

    @property
    def change(self) -> Optional[int]:
        """Calculate absolute change from previous run."""
        if self.previous_count is None:
            return None
        return self.current_count - self.previous_count

    @property
    def change_percent(self) -> Optional[float]:
        """Calculate percentage change from previous run."""
        if self.previous_count is None or self.previous_count == 0:
            return None
        return ((self.current_count - self.previous_count) / self.previous_count) * 100

    @property
    def trend_indicator(self) -> str:
        """Get visual trend indicator."""
        if self.change is None:
            return "➖"  # New test
        elif self.change > 0:
            return "📈"  # Increasing
        elif self.change < 0:
            return "📉"  # Decreasing
        else:
            return "➡️"  # No change


@dataclass
class QATestResult:
    """Enhanced test result with trend information."""

    qa_type: str
    rc_version: str
    rc_short: str  # RC1 or RC2
    layer_name: str
    test_name: str
    issue_type: str
    trend_data: TrendData
    run_count: int  # Number of runs for this test


class EnhancedQAConverter:
    """Enhanced version of FileGDBConverter with better statistics and trend analysis."""

    # RC version mappings
    RC_VERSION_MAP = {"2016-12-31": "RC1", "2030-12-31": "RC2"}

    # Test schedule mapping (day of week: 0=Monday, 6=Sunday)
    TEST_SCHEDULES = {
        "Topology": {
            "2030-12-31": 4,  # Friday for RC2
            "2016-12-31": 5,  # Saturday for RC1
        },
        "TechnicalQualityAssurance": {
            "2030-12-31": 4,  # Can be customized
            "2016-12-31": 5,
        },
    }

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """Initialize with existing DuckDB connection."""
        self.conn = conn

    def close(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()

    def get_enhanced_statistics_summary(
        self,
        qa_test_type: Optional[QATestType] = None,
        target_week: Optional[datetime] = None,
        target_date: Optional[datetime] = None,
        rc_versions: Optional[List[RCVersion]] = None,
        include_trends: bool = True,
        top_n: int = 20,
    ) -> Tuple[List[QATestResult], pd.DataFrame]:
        """
        Get enhanced QA statistics with trend analysis.

        Args:
            qa_test_type: Filter by QA test type ("Topology" or "TechnicalQualityAssurance")
            target_week: Get results for a specific week (uses Monday of that week)
            target_date: Get results for a specific date
            rc_versions: List of RC versions to include (defaults to both)
            include_trends: Whether to calculate trend comparisons
            top_n: Maximum number of results to return

        Returns:
            Tuple of (test_results_list, raw_dataframe)
        """

        # TODO
        console.print(f"Trend: {include_trends}")

        # Default to both RC versions if not specified
        if rc_versions is None:
            rc_versions = ["2016-12-31", "2030-12-31"]

        # Determine date range
        if target_week:
            # Find Monday of the target week
            monday = target_week - timedelta(days=target_week.weekday())
            start_date = monday
            end_date = monday + timedelta(days=6)  # Sunday
        elif target_date:
            start_date = target_date
            end_date = target_date + timedelta(days=1)
        else:
            # Default to last 7 days for latest runs
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

        logger.info(f"Querying QA stats from {start_date} to {end_date}")

        # Get current period results
        current_results = self._get_results_for_period(
            start_date=start_date,
            end_date=end_date,
            qa_test_type=qa_test_type,
            rc_versions=rc_versions,
        )

        if current_results.empty:
            logger.warning("No current results found for specified criteria")
            return [], pd.DataFrame()

        # Get trend data if requested
        test_results = []
        if include_trends:
            test_results = self._calculate_trends(
                current_results, qa_test_type, rc_versions
            )

            console.print(test_results)  # TODO
        else:
            # Create results without trends
            for _, row in current_results.iterrows():
                trend_data = TrendData(
                    current_count=row["total_count"],
                    previous_count=None,
                    current_date=row["latest_run"],
                    previous_date=None,
                )

                test_result = QATestResult(
                    qa_type=row["verification_type"],
                    rc_version=row["rc_version"],
                    rc_short=self.RC_VERSION_MAP.get(row["rc_version"], "RC?"),
                    test_name=row["test_name"],
                    issue_type=row["issue_type"],
                    trend_data=trend_data,
                    run_count=row["num_runs"],
                )
                test_results.append(test_result)

        # Sort by issue count (descending) and limit
        test_results.sort(key=lambda x: x.trend_data.current_count, reverse=True)
        test_results = test_results[:top_n]

        return test_results, current_results

    def _get_results_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        qa_test_type: Optional[str],
        rc_versions: List[str],
    ) -> pd.DataFrame:
        """Get test results for a specific time period."""

        query = """
            SELECT 
                s.verification_type,
                s.rc_version,
                ts.test_name,
                ts.issue_type,
                ts.layer_name, 
                CAST(SUM(ts.feature_count) AS INTEGER) as total_count,
                COUNT(DISTINCT s.id) as num_runs,
                MAX(s.timestamp) as latest_run,
                MIN(s.timestamp) as earliest_run
            FROM test_stats ts
            JOIN gdb_summaries s ON ts.gdb_summary_id = s.id
            WHERE s.timestamp >= ? AND s.timestamp < ?
        """

        params = [start_date, end_date]

        # Add QA test type filter
        if qa_test_type:
            query += " AND s.verification_type = ?"
            params.append(qa_test_type)

        # Add RC version filter
        if rc_versions:
            placeholders = ",".join("?" * len(rc_versions))
            query += f" AND s.rc_version IN ({placeholders})"
            params.extend(rc_versions)

        query += """
            GROUP BY s.verification_type, s.rc_version, ts.test_name, ts.issue_type, ts.layer_name
            ORDER BY total_count DESC
        """

        return self.conn.execute(query, params).df()

    def _calculate_trends(
        self,
        current_results: pd.DataFrame,
        qa_test_type: Optional[str],
        rc_versions: List[str],
    ) -> List[QATestResult]:
        """Calculate trend data by comparing with previous runs."""

        test_results = []

        for _, row in current_results.iterrows():
            # Get previous run data for this specific test/RC combination
            previous_data = self._get_previous_run_data(
                verification_type=row["verification_type"],
                rc_version=row["rc_version"],
                test_name=row["test_name"],
                issue_type=row["issue_type"],
                before_date=row["earliest_run"],
            )

            # Create trend data
            trend_data = TrendData(
                current_count=row["total_count"],
                previous_count=previous_data["count"] if previous_data else None,
                current_date=row["latest_run"],
                previous_date=previous_data["date"] if previous_data else None,
            )

            # Create test result
            test_result = QATestResult(
                qa_type=row["verification_type"],
                rc_version=row["rc_version"],
                rc_short=self.RC_VERSION_MAP.get(row["rc_version"], "RC?"),
                layer_name=row["layer_name"],
                test_name=row["test_name"],
                issue_type=row["issue_type"],
                trend_data=trend_data,
                run_count=row["num_runs"],
            )

            test_results.append(test_result)

        return test_results

    def _get_previous_run_data(
        self,
        verification_type: str,
        rc_version: str,
        test_name: str,
        issue_type: str,
        before_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent previous run data for a specific test."""

        query = """
            SELECT 
                CAST(SUM(ts.feature_count) AS INTEGER) as count,
                MAX(s.timestamp) as date
            FROM test_stats ts
            JOIN gdb_summaries s ON ts.gdb_summary_id = s.id
            WHERE s.verification_type = ?
                AND s.rc_version = ?
                AND ts.test_name = ?
                AND ts.issue_type = ?
                AND s.timestamp < ?
            GROUP BY s.verification_type, s.rc_version, ts.test_name, ts.issue_type
            ORDER BY date DESC
            LIMIT 1
        """

        result = self.conn.execute(
            query, [verification_type, rc_version, test_name, issue_type, before_date]
        ).fetchone()

        if result:
            return {"count": result[0], "date": result[1]}
        return None

    def display_enhanced_summary(
        self,
        test_results: List[QATestResult],
        title: str = "QA Test Results with Trends",
    ) -> None:
        """Display enhanced summary with trends in a rich table."""

        if not test_results:
            console.print("[yellow]No test results to display[/yellow]")
            return

        # Group by RC version for better organization
        rc1_results = [r for r in test_results if r.rc_short == "RC1"]
        rc2_results = [r for r in test_results if r.rc_short == "RC2"]

        # Display summary statistics first
        total_issues = sum(r.trend_data.current_count for r in test_results)
        unique_tests = len(set(r.test_name for r in test_results))

        stats_panel = Panel(
            f"[bold]Total Issues:[/bold] {total_issues:,}\n"
            f"[bold]Unique Tests:[/bold] {unique_tests}\n"
            f"[bold]RC1 Issues:[/bold] {sum(r.trend_data.current_count for r in rc1_results):,}\n"
            f"[bold]RC2 Issues:[/bold] {sum(r.trend_data.current_count for r in rc2_results):,}",
            title="📊 Summary Statistics",
            border_style="blue",
        )
        console.print(stats_panel)

        # Create main results table
        table = Table(title=title)
        table.add_column("QA Type", style="cyan", width=12)
        table.add_column("RC", style="bold", width=4)
        table.add_column("Test Name", max_width=25)
        table.add_column("Issue Type", width=12)
        table.add_column("Current", justify="right", style="bold")
        table.add_column("Previous", justify="right", style="dim")
        table.add_column("Change", justify="right")
        table.add_column("Trend", justify="center", width=4)
        table.add_column("Latest Run", style="dim", width=12)

        for result in test_results:
            # Style based on issue type
            if "error" in result.issue_type.lower():
                issue_style = "bold red"
            elif "warning" in result.issue_type.lower():
                issue_style = "bold yellow"
            else:
                issue_style = "dim"

            # Format change column
            change_text = "New"
            change_style = "cyan"
            if result.trend_data.change is not None:
                change_value = result.trend_data.change
                if change_value > 0:
                    change_text = f"+{change_value:,}"
                    change_style = "red"
                elif change_value < 0:
                    change_text = f"{change_value:,}"
                    change_style = "green"
                else:
                    change_text = "0"
                    change_style = "dim"

                # Add percentage if available
                if result.trend_data.change_percent is not None:
                    pct = result.trend_data.change_percent
                    change_text += f" ({pct:+.1f}%)"

            table.add_row(
                result.qa_type.replace("TechnicalQualityAssurance", "TQA"),
                result.rc_short,
                result.test_name,
                f"[{issue_style}]{result.issue_type}[/{issue_style}]",
                f"{result.trend_data.current_count:,}",
                f"{result.trend_data.previous_count:,}"
                if result.trend_data.previous_count
                else "—",
                f"[{change_style}]{change_text}[/{change_style}]",
                result.trend_data.trend_indicator,
                result.trend_data.current_date.strftime("%m-%d %H:%M"),
            )

        console.print(table)

    def get_test_schedule_info(self, qa_test_type: QATestType) -> Dict[str, str]:
        """Get scheduling information for tests."""
        schedule_info = {}

        for rc_version, day_num in self.TEST_SCHEDULES.get(qa_test_type, {}).items():
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            rc_short = self.RC_VERSION_MAP.get(rc_version, "RC?")
            schedule_info[rc_short] = day_names[day_num]

        return schedule_info

    def get_weekly_summary(
        self, qa_test_type: QATestType, weeks_back: int = 4
    ) -> pd.DataFrame:
        """Get a weekly summary showing trends over multiple weeks."""

        weekly_results = []
        end_date = datetime.now()

        for week_offset in range(weeks_back):
            # Calculate week start (Monday)
            week_start = end_date - timedelta(
                days=end_date.weekday() + (week_offset * 7)
            )
            week_end = week_start + timedelta(days=6)

            # Get results for this week
            results = self._get_results_for_period(
                start_date=week_start,
                end_date=week_end + timedelta(days=1),  # Include Sunday
                qa_test_type=qa_test_type,
                rc_versions=["2016-12-31", "2030-12-31"],
            )

            if not results.empty:
                # Aggregate by RC version
                rc_summary = (
                    results.groupby("rc_version")
                    .agg({"total_count": "sum", "num_runs": "sum", "latest_run": "max"})
                    .reset_index()
                )

                for _, row in rc_summary.iterrows():
                    weekly_results.append(
                        {
                            "week_start": week_start.strftime("%Y-%m-%d"),
                            "week_number": f"Week -{week_offset:02d}",
                            "rc_version": row["rc_version"],
                            "rc_short": self.RC_VERSION_MAP.get(
                                row["rc_version"], "RC?"
                            ),
                            "total_issues": row["total_count"],
                            "test_runs": row["num_runs"],
                            "last_run": row["latest_run"],
                        }
                    )

        return pd.DataFrame(weekly_results)


# Enhanced CLI command integration
def enhanced_stats_command(
    qa_type: Optional[str] = None,
    target_week: Optional[str] = None,
    target_date: Optional[str] = None,
    rc_versions: Optional[str] = None,
    show_trends: bool = True,
    show_schedule: bool = False,
    weekly_summary: bool = False,
    top_n: int = 20,
):
    """
    Enhanced stats command with trend analysis.

    Usage examples:
        enhanced_stats_command(qa_type="Topology", show_trends=True)
        enhanced_stats_command(target_week="2025-01-20", rc_versions="RC2")
        enhanced_stats_command(weekly_summary=True, qa_type="TechnicalQualityAssurance")
    """

    # Parse inputs
    parsed_target_week = None
    if target_week:
        try:
            parsed_target_week = datetime.strptime(target_week, "%Y-%m-%d")
        except ValueError:
            console.print(
                f"[red]Invalid week format: {target_week}. Use YYYY-MM-DD[/red]"
            )
            return

    parsed_target_date = None
    if target_date:
        try:
            parsed_target_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            console.print(
                f"[red]Invalid date format: {target_date}. Use YYYY-MM-DD[/red]"
            )
            return

    # Parse RC versions
    parsed_rc_versions = None
    if rc_versions:
        rc_map = {"RC1": "2016-12-31", "RC2": "2030-12-31"}
        parsed_rc_versions = []
        for rc in rc_versions.split(","):
            rc = rc.strip().upper()
            if rc in rc_map:
                parsed_rc_versions.append(rc_map[rc])
            else:
                console.print(f"[red]Invalid RC version: {rc}. Use RC1 or RC2[/red]")
                return

    # This would be initialized with your existing database connection
    # conn = duckdb.connect("path/to/your/verification_stats.duckdb")
    # enhanced_converter = EnhancedQAConverter(conn)

    console.print("[dim]Enhanced QA statistics would be displayed here[/dim]")
    console.print(
        f"[dim]Filters: QA Type={qa_type}, Week={target_week}, Date={target_date}[/dim]"
    )

    # Example of how to use:
    # results, df = enhanced_converter.get_enhanced_statistics_summary(
    #     qa_test_type=qa_type,
    #     target_week=parsed_target_week,
    #     target_date=parsed_target_date,
    #     rc_versions=parsed_rc_versions,
    #     include_trends=show_trends,
    #     top_n=top_n
    # )
    #
    # enhanced_converter.display_enhanced_summary(results)
    #
    # if show_schedule and qa_type:
    #     schedule = enhanced_converter.get_test_schedule_info(qa_type)
    #     console.print(f"\n[bold]Test Schedule for {qa_type}:[/bold]")
    #     for rc, day in schedule.items():
    #         console.print(f"  {rc}: {day}")
    #
    # if weekly_summary:
    #     weekly_df = enhanced_converter.get_weekly_summary(qa_type, weeks_back=4)
    #     # Display weekly trends...


if __name__ == "__main__":
    # Example usage
    enhanced_stats_command(
        qa_type="Topology", show_trends=True, show_schedule=True, top_n=15
    )
