# Parcels Data Dictionary

| Project column | Description |
| --- | --- |
| PID | Unique 10-digit parcel identifier (ward, parcel, and sub-parcel components). |
| CONDO_ID | Condo main parcel ID that groups related condo units. |
| NUM_BLDGS | Number of buildings associated with the parcel record. |
| LU | Land use category/class label. |
| LU_DESC | Land use description text. |
| BLDG_TYPE | Building type or structural style classification. |
| RES_FLOOR | Number of residential floors. |
| RES_UNITS | Count of residential units. |
| TT_RMS | Total number of rooms. |
| BED_RMS | Number of bedrooms. |
| FULL_BTH | Number of full bathrooms. |
| HLF_BTH | Number of half bathrooms. |
| KITCHENS | Number of kitchens. |
| OVERALL_COND | Overall condition rating. |
| INT_COND | Interior condition rating. |
| EXT_COND | Exterior condition rating. |
| NUM_PARKING | Number of parking spaces. |
| STRUCTURE_CLASS | Structure or construction class. |
| YR_REMODEL | Year of most recent major remodel. |
| YR_BUILT | Year built. |
| LAND_VALUE | Assessed land value. |
| BLDG_VALUE | Assessed building/improvement value. |
| TOTAL_VALUE | Total assessed value. |
| LAND_SF | Land area in square feet. |
| GROSS_AREA | Gross building area. |
| LIVING_AREA | Living area in square feet. |
| zoning_use | Zoning subdistrict/use designation joined from Boston zoning subdistrict polygons. |
| max_far | Maximum floor area ratio allowed by zoning for the intersecting subdistrict. |
| max_height | Maximum building height allowed by zoning for the intersecting subdistrict. |
| front_setback | Minimum required front setback from the lot line in the intersecting zoning subdistrict. |
| side_setback | Minimum required side setback from the lot line in the intersecting zoning subdistrict. |
| rear_setback | Minimum required rear setback from the lot line in the intersecting zoning subdistrict. |
| max_dua | Maximum dwelling units per area metric allowed by zoning (as provided by source field). |
| max_floors | Maximum number of stories/floors allowed by zoning for the intersecting subdistrict. |
| median_hh_income | Tract-level median household income from ACS 5-year table B19013_001E, spatially joined to parcels by tract. |
| emp_dist_m | Straight-line distance in meters from parcel representative point to the nearest employment center/CBD. |
| neighborhood_walkability | Network-based walkability score (0-100) computed from OSM walking-network distance to nearby daily destinations (grocery, food, education, parks, transit); category scores decay linearly to 0 at 1,600 m, and the parcel score is the mean of available category scores. |
| geometry | Parcel geometry in WKT format. |

## LU Code Categories

| LU code | Category |
| --- | --- |
| A | Residential 7 or more units |
| AH | Agricultural/Horticultural |
| C | Commercial |
| CC | Commercial condominium |
| CD | Residential condominium unit |
| CL | Commercial land |
| CM | Condominium main (physical structure housing all related condo units with no assessed value) |
| CP | Condo parking |
| E | Tax-exempt |
| EA | Tax-exempt (121A) |
| I | Industrial |
| R1 | Residential 1-family |
| R2 | Residential 2-family |
| R3 | Residential 3-family |
| R4 | Residential 4 or more family |
| RC | Mixed use (residential and commercial) |
| RL | Residential land |