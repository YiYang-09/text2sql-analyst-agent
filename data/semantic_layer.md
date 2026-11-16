# Semantic layer: EU lorry registrations (Eurostat)

Lorries over 3.5 t only (not road tractors, not vans). Years 2013 to 2025. 2025 is incomplete.
"Electric" means battery-electric (fuel code `ELC`). "New lorries" means first registrations in a year;
"stock" or "lorries in use" means the fleet on the road at the end of the year.

## Business rules

1. **Do not double count countries.** `dim_country` and the fact tables contain an aggregate row
   `EU27_2020`. When summing over countries, exclude it (`is_aggregate_geo = false` on facts,
   `is_aggregate = false` on `dim_country`). For EU27 totals use `metrics_eu27_year`.
2. **EU27 means the 27 member states** (`dim_country.is_eu27 = true`). Norway, Switzerland, Iceland,
   the UK and candidate countries are in the data but are not EU27.
3. **Do not double count fuels.** `dim_fuel.fuel_level` is `total`, `subtotal` or `leaf`. Sum units over
   one level only: `fuel_level = 'total'` for all lorries, `fuel_level = 'leaf'` for a fuel breakdown.
   `PET`, `DIE`, `ALT`, `CNG`, `LNG` are subtotals and overlap with the leaves.
4. **Missing is not zero.** Null values mean the country did not report. Do not treat them as 0.
   A share is only defined when both numerator and denominator exist.
5. **Shares are fractions** between 0 and 1 (`0.0412` means 4.12 %).
6. **2025 is partial.** Only 25 of 27 EU countries reported new lorries for 2025 and there is no EU27
   total for 2025 (`metrics_eu27_year` has nulls for 2025). Say so when a question involves 2025.
7. **Prefer the metric tables.** `metrics_country_year` and `metrics_eu27_year` already contain
   electric counts, totals, shares and year-on-year change with one definition each. Use the fact
   tables only for fuels other than electric or for the fuel hierarchy.
8. **Country names.** Greece is `EL` and the United Kingdom is `UK` in `geo`; `iso2` holds `GR` and `GB`.

## Tables

### dim_country (one row per geography)
- `geo` (text, key): Eurostat code, for example `DE`, `EL`, `EU27_2020`.
- `country_name` (text): English name.
- `is_aggregate` (bool): true only for `EU27_2020`.
- `is_eu27` (bool): member of the EU27.
- `is_nordic` (bool): SE, DK, FI, NO, IS.
- `iso2` (text): ISO 3166 alpha-2 code.

### dim_fuel (one row per fuel code)
- `mot_nrg` (text, key): Eurostat motor energy code, for example `TOTAL`, `ELC`, `DIE_X_HYB`.
- `fuel_name` (text): Eurostat label.
- `fuel_group` (text): readable group: Total, Battery electric, Diesel, Diesel hybrid, Petrol,
  Petrol hybrid, Gas, Other alternative, Subtotal.
- `is_parent` (bool): true for total and subtotal codes.
- `fuel_level` (text): `total`, `subtotal` or `leaf`.

### dim_year
- `year` (int, key), `is_partial_year` (bool, true for 2025).

### dim_flag
- `flag` (text, key): `b` break in series, `e` estimated, `i` see metadata (low reliability),
  `m` missing, `p` provisional. `meaning` (text).

### fct_new_lorries_bi (new registrations, one row per geo, fuel code, year)
- `geo`, `mot_nrg`, `year`, `units` (int, may be null), `flag` (text, may be null),
  `is_aggregate_geo` (bool), `fuel_level` (text, copied from dim_fuel).

### fct_stock_lorries_bi (lorries in use, same columns as fct_new_lorries_bi)

### metrics_country_year (one row per geo and year, countries and EU27_2020 aggregate row)
- `new_total`, `new_electric`: new lorries, all fuels and battery-electric.
- `stock_total`, `stock_electric`: lorries in use, all fuels and battery-electric.
- `new_electric_share` = new_electric / new_total; `stock_electric_share` = stock_electric / stock_total.
- `new_small_denominator`, `stock_small_denominator` (bool): fewer than 100 lorries; shares are unstable.
- `new_electric_change`: new_electric minus previous year; `new_electric_yoy`: relative change
  (0.5 = +50 %). Both are null unless the previous year exists and had at least 50 electric lorries.
- `*_flag` columns: Eurostat flags of the underlying values.
- `new_low_reliability`, `stock_low_reliability` (bool): a flag `b` or `i` on the inputs.

### metrics_eu27_year (one row per year, EU27 rebuilt from the 27 member states)
- `n_countries_new`, `n_countries_stock`: how many members reported both total and electric.
- `is_complete_new`, `is_complete_stock` (bool): all 27 reported.
- `new_total`, `new_electric`, `new_electric_share`, `stock_total`, `stock_electric`,
  `stock_electric_share`: null unless all 27 reported.
- `partial_*` columns: sums over the countries that did report (for reference only).

### fuel_mix_new (new lorries by country, year and fuel_group; countries only)
- `geo`, `year`, `fuel_group`, `units`. Includes an `Unclassified` group: the part of the total
  without a fuel breakdown.

### fuel_mix_quality (one row per country and year)
- `total_units`, `classified_units`, `classified_share`, `is_usable_for_mix` (bool, classified_share
  between 0.95 and 1). Use only usable rows for fuel-mix comparisons.
