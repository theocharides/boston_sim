"""Tests that validate the content and shape of preprocessed parcel data."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_CSV = REPO_ROOT / "outputs" / "parcels_preprocessed.csv"

# Known valid LU codes from the project data dictionary.
KNOWN_LU_CODES = {
    "A", "AH", "C", "CC", "CD", "CL", "CM", "CP", "E", "EA",
    "I", "R1", "R2", "R3", "R4", "RC", "RL",
}
# Sentinel used by add_income.py for missing ACS income.
INCOME_MISSING_SENTINEL = -666666666.0

# Maximum tolerated missing share for critical modeling columns.
MAX_CRITICAL_MISSING_SHARE = 0.10

# Columns expected after the full pipeline has run (clean + zoning + neighborhood + income + emp_dist).
REQUIRED_COLUMNS: list[str] = [
    "PID",
    "LU",
    "LU_DESC",
    "TOTAL_VALUE",
    "LAND_VALUE",
    "BLDG_VALUE",
    "LAND_SF",
    "LIVING_AREA",
    "RES_UNITS",
    "BLDG_TYPE",
    "YR_BUILT",
    "INT_COND",
    "zoning_use",
    "neighborhood_name",
    "median_hh_income",
    "emp_dist_m",
    "geometry",
]

# Columns added only by optional steps — checked for presence but not required values.
OPTIONAL_COLUMNS: list[str] = [
    "max_far",
    "max_height",
    "max_dua",
    "max_floors",
    "front_setback",
    "side_setback",
    "rear_setback",
    "neighborhood_id",
    "GROSS_AREA",
]

# Columns used by the hedonic model that must remain mostly populated.
CRITICAL_MODELING_COLUMNS: list[str] = [
    "TOTAL_VALUE",
    "LAND_SF",
    "LIVING_AREA",
    "INT_COND",
    "zoning_use",
    "neighborhood_name",
    "median_hh_income",
    "emp_dist_m",
]


def load_preprocessed_parcels() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_CSV, low_memory=False)


def get_yr_built_outliers(parcels_df: pd.DataFrame) -> pd.Series:
    yr_built = pd.to_numeric(parcels_df["YR_BUILT"], errors="coerce")
    return yr_built[(yr_built < 1600) | (yr_built > 2030)].dropna()


@pytest.fixture(scope="module")
def parcels() -> pd.DataFrame:
    if not PREPROCESSED_CSV.exists():
        pytest.skip(f"Preprocessed parcel CSV not found: {PREPROCESSED_CSV}")
    return load_preprocessed_parcels()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_required_columns_present(self, parcels: pd.DataFrame) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in parcels.columns]
        assert missing == [], f"Required columns missing from preprocessed output: {missing}"

    def test_optional_columns_present(self, parcels: pd.DataFrame) -> None:
        missing = [col for col in OPTIONAL_COLUMNS if col not in parcels.columns]
        assert missing == [], f"Optional pipeline columns missing (was a step skipped?): {missing}"

    def test_no_duplicate_column_names(self, parcels: pd.DataFrame) -> None:
        dupes = [col for col in set(parcels.columns) if list(parcels.columns).count(col) > 1]
        assert dupes == [], f"Duplicate column names in output: {dupes}"


# ---------------------------------------------------------------------------
# Row count and uniqueness
# ---------------------------------------------------------------------------

class TestRowsAndIdentifiers:
    def test_has_rows(self, parcels: pd.DataFrame) -> None:
        assert len(parcels) > 0, "Preprocessed parcel table is empty."

    def test_pid_is_unique(self, parcels: pd.DataFrame) -> None:
        dupes = parcels["PID"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate PID values found."

    def test_pid_not_null(self, parcels: pd.DataFrame) -> None:
        nulls = parcels["PID"].isna().sum()
        assert nulls == 0, f"{nulls} null PID values found."

    def test_pid_is_numeric_string_10_digits(self, parcels: pd.DataFrame) -> None:
        pid_str = parcels["PID"].astype(str)
        invalid = pid_str[~pid_str.str.fullmatch(r"\d{1,10}")]
        assert len(invalid) == 0, (
            f"{len(invalid)} PID values are not numeric or exceed 10 digits: "
            f"{invalid.head(5).tolist()}"
        )


# ---------------------------------------------------------------------------
# Land use
# ---------------------------------------------------------------------------

class TestLandUse:
    def test_lu_not_all_null(self, parcels: pd.DataFrame) -> None:
        null_share = parcels["LU"].isna().mean()
        assert null_share < 0.5, f"More than 50% of LU values are null ({null_share:.1%})."

    def test_lu_values_are_known_codes(self, parcels: pd.DataFrame) -> None:
        non_null = parcels["LU"].dropna().astype(str).str.strip()
        # Strip compound codes like "RL - RL" (source data sometimes includes description)
        parsed = non_null.str.split(r"\s*-\s*", expand=False).str[0].str.strip()
        unknown = set(parsed.unique()) - KNOWN_LU_CODES
        assert unknown == set(), (
            f"Unexpected LU code values (not in data dictionary): {unknown}"
        )

    def test_residential_lu_rows_exist(self, parcels: pd.DataFrame) -> None:
        residential = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}
        lu_clean = (
            parcels["LU"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.split(r"\s*-\s*", expand=False)
            .str[0]
            .str.strip()
        )
        count = lu_clean.isin(residential).sum()
        assert count > 0, "No residential LU code rows found in preprocessed output."


# ---------------------------------------------------------------------------
# Value columns
# ---------------------------------------------------------------------------

class TestValueColumns:
    def test_total_value_not_all_null(self, parcels: pd.DataFrame) -> None:
        null_share = pd.to_numeric(parcels["TOTAL_VALUE"], errors="coerce").isna().mean()
        assert null_share < 0.5, f"More than 50% of TOTAL_VALUE is null ({null_share:.1%})."

    def test_total_value_non_negative(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["TOTAL_VALUE"], errors="coerce").dropna()
        negatives = (vals < 0).sum()
        assert negatives == 0, f"{negatives} rows have negative TOTAL_VALUE."

    def test_land_value_non_negative(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["LAND_VALUE"], errors="coerce").dropna()
        negatives = (vals < 0).sum()
        assert negatives == 0, f"{negatives} rows have negative LAND_VALUE."

    def test_bldg_value_non_negative(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["BLDG_VALUE"], errors="coerce").dropna()
        negatives = (vals < 0).sum()
        assert negatives == 0, f"{negatives} rows have negative BLDG_VALUE."

    def test_total_value_matches_sum_of_components(self, parcels: pd.DataFrame) -> None:
        land = pd.to_numeric(parcels["LAND_VALUE"], errors="coerce").fillna(0)
        bldg = pd.to_numeric(parcels["BLDG_VALUE"], errors="coerce").fillna(0)
        total = pd.to_numeric(parcels["TOTAL_VALUE"], errors="coerce")
        both_present = total.notna() & (land > 0) & (bldg > 0)
        if both_present.sum() == 0:
            pytest.skip("No rows with both LAND_VALUE and BLDG_VALUE present.")
        expected = (land + bldg)[both_present]
        actual = total[both_present]
        mismatches = (abs(actual - expected) > 1.0).sum()
        mismatch_rate = mismatches / both_present.sum()
        # Assessor data for condo aggregations legitimately rounds to a different total;
        # 15% is a generous ceiling — above that suggests a pipeline problem.
        assert mismatch_rate < 0.15, (
            f"{mismatch_rate:.1%} of rows have TOTAL_VALUE != LAND_VALUE + BLDG_VALUE "
            f"(tolerance $1; expected <15%)."
        )

    def test_land_sf_positive_where_present(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["LAND_SF"], errors="coerce").dropna()
        non_positive = (vals <= 0).sum()
        assert non_positive == 0, f"{non_positive} rows have non-positive LAND_SF."

    def test_living_area_non_negative_where_present(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["LIVING_AREA"], errors="coerce").dropna()
        negatives = (vals < 0).sum()
        assert negatives == 0, f"{negatives} rows have negative LIVING_AREA."

    def test_yr_built_plausible(self, parcels: pd.DataFrame) -> None:
        yr_outliers = get_yr_built_outliers(parcels)
        yr = pd.to_numeric(parcels["YR_BUILT"], errors="coerce").dropna()
        if len(yr) == 0:
            pytest.skip("No non-null YR_BUILT values.")
        out_of_range = len(yr_outliers)
        # A small number of data-entry errors (e.g. 20198 instead of 2019) are
        # tolerated, but more than 10 such rows suggests a systemic cleaning issue.
        if out_of_range > 0:
            sample_outliers = yr_outliers.head(5).tolist()
            warnings.warn(
                "WARNING: YR_BUILT outliers found outside [1600, 2030]: "
                f"count={out_of_range}, sample={sample_outliers}",
                stacklevel=1,
            )
        assert out_of_range <= 10, (
            f"{out_of_range} rows have YR_BUILT outside [1600, 2030] "
            f"(known data-entry errors tolerated up to 10)."
        )


# ---------------------------------------------------------------------------
# Locational enrichment
# ---------------------------------------------------------------------------

class TestLocationalEnrichment:
    def test_emp_dist_m_present_and_positive(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["emp_dist_m"], errors="coerce")
        null_share = vals.isna().mean()
        assert null_share < 0.5, f"More than 50% of emp_dist_m is null ({null_share:.1%})."
        non_null = vals.dropna()
        non_positive = (non_null <= 0).sum()
        assert non_positive == 0, f"{non_positive} rows have non-positive emp_dist_m."

    def test_neighborhood_name_mostly_present(self, parcels: pd.DataFrame) -> None:
        null_share = parcels["neighborhood_name"].isna().mean()
        assert null_share < 0.5, (
            f"More than 50% of neighborhood_name is null ({null_share:.1%}). "
            "Was add_neighborhood.py run?"
        )

    def test_median_hh_income_mostly_present(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["median_hh_income"], errors="coerce")
        # Exclude the sentinel value used for missing ACS data.
        real_missing = (vals.isna() | (vals == INCOME_MISSING_SENTINEL)).mean()
        assert real_missing < 0.5, (
            f"More than 50% of median_hh_income is missing or sentinel ({real_missing:.1%}). "
            "Was add_income.py run?"
        )

    def test_median_hh_income_no_unexpected_negatives(self, parcels: pd.DataFrame) -> None:
        vals = pd.to_numeric(parcels["median_hh_income"], errors="coerce").dropna()
        # Only flag values more negative than the known sentinel (which is -666666666).
        extremely_negative = (vals < INCOME_MISSING_SENTINEL).sum()
        assert extremely_negative == 0, (
            f"{extremely_negative} rows have median_hh_income more negative than the sentinel value."
        )

    def test_zoning_use_mostly_present(self, parcels: pd.DataFrame) -> None:
        null_share = parcels["zoning_use"].isna().mean()
        assert null_share < 0.5, (
            f"More than 50% of zoning_use is null ({null_share:.1%}). "
            "Was add_zoning.py run?"
        )


# ---------------------------------------------------------------------------
# Critical column completeness
# ---------------------------------------------------------------------------

class TestCriticalColumnCompleteness:
    @staticmethod
    def _residential_subset(parcels: pd.DataFrame) -> pd.DataFrame:
        residential = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}
        lu_clean = (
            parcels["LU"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.split(r"\s*-\s*", expand=False)
            .str[0]
            .str.strip()
            .str.upper()
        )
        return parcels.loc[lu_clean.isin(residential)].copy()

    @pytest.mark.parametrize("column", CRITICAL_MODELING_COLUMNS)
    def test_critical_columns_under_10pct_missing_on_residential_rows(
        self,
        parcels: pd.DataFrame,
        column: str,
    ) -> None:
        residential = self._residential_subset(parcels)
        if residential.empty:
            pytest.skip("No residential rows available for completeness checks.")

        if column == "median_hh_income":
            vals = pd.to_numeric(residential[column], errors="coerce")
            missing_share = (vals.isna() | (vals == INCOME_MISSING_SENTINEL)).mean()
        else:
            missing_share = residential[column].isna().mean()

        assert missing_share < MAX_CRITICAL_MISSING_SHARE, (
            f"{column} missing share is {missing_share:.1%} on residential rows; "
            f"expected < {MAX_CRITICAL_MISSING_SHARE:.0%}."
        )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_geometry_mostly_present(self, parcels: pd.DataFrame) -> None:
        null_share = parcels["geometry"].isna().mean()
        assert null_share < 0.1, (
            f"More than 10% of geometry values are null ({null_share:.1%})."
        )

    def test_geometry_looks_like_wkt(self, parcels: pd.DataFrame) -> None:
        non_null = parcels["geometry"].dropna().astype(str)
        if len(non_null) == 0:
            pytest.skip("No non-null geometry values.")
        wkt_pattern = re.compile(r"^(POLYGON|MULTIPOLYGON|POINT|LINESTRING)", re.IGNORECASE)
        invalid = non_null[~non_null.str.match(wkt_pattern)]
        assert len(invalid) == 0, (
            f"{len(invalid)} geometry values do not look like WKT: {invalid.head(3).tolist()}"
        )
