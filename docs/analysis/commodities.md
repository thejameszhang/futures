# Commodity Inventory Comcodes

This document explains the Datastream `comcode` inventory series mapped in `tier1.yaml` and
`tier2.yaml`. These are inventory/stock measures, not prices. They should not be coalesced or
summed unless explicitly stated. Most series come from `DSCmExtVal` with item `IV` (inventory
volume). Some series (WASDE ending stocks, Rotterdam vegetable oil stocks, and a few legacy
inventories) are sourced from `DSCmVal` (close_) because item `IV` is not populated for those
comcodes.

## How to Interpret `value_` Units

`value_` is reported in the native units of each Datastream series. Units are not standardized across
commodities. Use `dscminfo.csv` to interpret units:
- `unitcode` maps to a unit description via `dscmdesc.csv` (Type_ = 4).
- Many series also embed units in `name_` or `dsname` (e.g., MBBL, metric tonne, bales, bags, TWh).
- `isocur` is the currency code and is generally **not** relevant for inventory quantities unless the
  series is expressed in value terms (rare for these comcodes).

Examples:
- EIA petroleum series typically use **million barrels (MBBL)** or **thousand barrels** as stated in the
  series name.
- LME warehouse stocks use **metric tonnes**.
- ICE cotton stocks are in **bales**; ICE cocoa stocks often report **bags**.
- WASDE ending stocks are in **million pounds**, **thousand short tons**, or **million bales** as stated
  in the series name.
- Natural gas storage series use **BCF** or **TWh** depending on the region/series.

Do not mix series with different units without explicit conversion.

## Primary Inventory Series (tier1.yaml)

To keep the inventory characteristic interpretable and unit-consistent, we select
one primary inventory series per symbol for `tier1.yaml`. Selection rules:
deliverable or contract-relevant inventory when available, and stock *levels* (not
flows) for storage series. All other series remain documented below for alternative
specifications.

- ZW: `17301` (Chicago total wheat stocks). Delivery-point total inventory; broader
  than deliverable grades alone.
- KE: `17266` (Kansas City total HRW wheat stocks). Delivery-point total inventory.
- ZC: `17255` (Chicago total corn stocks). Delivery-point total inventory.
- ZS: `17283` (Chicago total soybean stocks). Delivery-point total inventory.
- CT: `13934` (ICE Europe certified bales stocks). Deliverable inventory proxy.
- CL: `15968` (Cushing, OK crude stocks). WTI delivery-point inventory.
- RB: `1573` (Total U.S. gasoline stocks). Broad gasoline inventory; aligns with
  the legacy HU series to preserve continuity through splicing.
- NG: `8033` (Working gas in storage). Inventory level; avoids flow series.

## Grains

### Chicago Wheat (ZW)
- `17301` (WHCHTTL): Chicago total wheat stocks. Use for delivery-point total stocks.
- `17300` (WHCHSRW): Soft Red Winter wheat deliverable grades at Chicago. Use for SRW deliverable supply.

### Kansas Wheat (KE)
- `17266` (HRKATTL): Kansas City total HRW wheat stocks. Use for KC delivery-point totals.
- `17249` (CMHRWST): HRW wheat total stocks across locations. Use for broad HRW inventory.

### Corn (ZC)
- `17248` (CMCORST): Total corn stocks across locations. Use for broad inventory.
- `17255` (CRCHTTL): Chicago total corn stocks. Use for delivery-point inventory.

### Soybeans (ZS)
- `17252` (CMSOYST): Total soybean stocks across locations. Use for broad inventory.
- `17283` (SYCHTTL): Chicago total soybean stocks. Use for delivery-point inventory.

### Oats (ZO)
- `17250` (CMOATST): CME oats total stocks. Use for broad inventory.
- `17243` (OACHTTL): CME oats Chicago total stocks. Use for delivery-point inventory.
- `17272` (OACHDGR): CME oats Chicago deliverable grades stocks.
- `19359` (OACHNGR): CME oats Chicago non-deliverable grades stocks.
- `17275` (OADUTTL): CME oats Duluth Superior total stocks.
- `17273` (OADUDGR): CME oats Duluth Superior deliverable grades stocks.
- `17274` (OADUNGR): CME oats Duluth Superior non-deliverable grades stocks.
- `17278` (OAMITTL): CME oats Minneapolis total stocks.
- `17276` (OAMIDGR): CME oats Minneapolis deliverable grades stocks.
- `17277` (OAMINGR): CME oats Minneapolis non-deliverable grades stocks.

### Rough Rice (ZR)
- `17251` (CMRICST): CME rice total stocks. Use for broad inventory.
- `17281` (RIARTTL): CME rice Arkansas total stocks. Delivery-region aggregate.
- `17279` (RIARDGR): CME rice Arkansas deliverable grades stocks.
- `17280` (RIARNGR): CME rice Arkansas non-deliverable grades stocks.

## Oilseeds & Oils

### Soybean Meal (ZM)
- `19006` (SMSTKM0): WASDE soybean meal ending stocks (month0). Broad balance-sheet inventory.

### Soybean Oil (ZL)
- `19013` (SOSTKM0): WASDE soybean oil ending stocks (month0). Broad balance-sheet inventory.

### Palm Oil (P)
- `3963` (VGOPALM): Rotterdam palm oil stocks (metric tonnes). Port inventory proxy.

## Softs

### Coffee (KC)
- `14009` (KCTOTTT): ICE coffee certified warehouse stocks total. Use for deliverable inventory.

### Cocoa (CC)
- `13874` (CCTOTAL): ICE cocoa certified warehouse stocks total. Use for deliverable inventory.

### Cocoa (C)
- `13874` (CCTOTAL): Same ICE cocoa certified warehouse stocks total as CC. Treat as deliverable inventory.

### Cotton (CT)
These are ICE Europe stock status buckets. Treat each status as a separate series.
- `13933` (CTAWTST): Awaiting review stocks. Use to track pipeline awaiting certification.
- `13934` (CTCRTST): Certified bales stocks. Use for deliverable inventory.
- `13935` (CTDCTST): Decertified stocks. Use as a measure of non-deliverable inventory.
- `13936` (CTISDST): Issued bales stocks. Use to track issued inventory within the certified system.

### Cotton #1 (CF)
- `18991` (CTSTKM0): WASDE cotton ending stocks (month0, million 480-lb bales). Broad balance-sheet inventory.

## Energy

### WTI Crude Oil (CL)
- `6406` (EIA1533): Total U.S. crude oil stocks. Headline national inventory level.
- `15968` (CRDOKL2): Cushing, OK crude stocks. Direct delivery-point inventory for WTI.
- `6405` (EIA1532): U.S. crude stocks excluding SPR. Commercial inventories.
- `6404` (EIA1531): U.S. crude stocks in the SPR. Strategic, policy-driven inventory.

Use Cushing for delivery-point analysis; use total or commercial for broad supply. Do not add
SPR to commercial unless you explicitly want total stocks (already provided by `6406`).

### RBOB Gasoline (RB)
- `1573` (EIAGSST): Total U.S. gasoline stocks. Broad inventory.
- `1546` (EIAFGST): Finished gasoline stocks. Refined end-product inventory.

### Unleaded Gasoline (HU)
- `1573` (EIAGSST): Total U.S. gasoline stocks. Broad inventory.

### Heating Oil / ULSD (HO)
- `1538` (EIADIST): Total U.S. distillate stocks. Use as heating oil inventory proxy.

### Gas Oil (G)
- `13814` (STKGOAA): ARA gasoil stocks (Amsterdam/Rotterdam/Antwerp). Use for European gasoil inventories.

### Natural Gas (NG)
These are storage level and activity series from `DSCmVal`.
- `8033` (EIA1669): Working gas in storage. Use as available storage level.
- `8032` (EIA1668): Base gas in storage. Structural inventory; not withdrawable.
- `8035` (EIA1671): Storage activity withdrawals. Flow series.
- `8037` (EIA1673): Net storage activity. Flow series.
- `8004` (EIA1640): Net storage withdrawals. Flow series.

Do not mix levels (working/base) with flows (withdrawals/net activity) in the same analysis.

### Propane (A7E, A9N)
- `1628` (EIAPRST): EIA U.S. propane/propylene end stocks (MBBL). Broad inventory proxy.

### Thermal Coal (TC)
- `13539` (SHQHACO): Qinhuangdao port coal inventory (tons). China port inventory proxy.

## Metals

### LME Base Metals
- `2064` (LADWARE): LME aluminium warehouse stocks total.
- `2093` (LCPWARE): LME copper warehouse stocks total.
- `2127` (LNIWARE): LME nickel warehouse stocks total.
- `2109` (LLDWARE): LME lead warehouse stocks total.
- `2170` (LZZWARE): LME zinc warehouse stocks total.
- `2153` (LTIWARE): LME tin warehouse stocks total.

Use for deliverable LME inventory. These are warehouse totals, not prices.

### COMEX Metals
- `6800` (GCCMXST): COMEX gold stocks total.
- `6824` (SICMXST): COMEX silver stocks total.
- `6805` (HGCMXST): COMEX copper stocks total.
- `6821` (PLNYMST): COMEX platinum stocks total.
- `13746` (NPAWARE): NYMEX palladium stocks total.

Use for COMEX deliverable inventory. Do not combine with LME series; they are different venues.

### SHFE Base Metals
- `5186` (CHALWHS): SHFE aluminium warehouse stocks total.
- `5195` (CHCUWHS): SHFE copper warehouse stocks total.
- `5228` (CHZNWHS): SHFE zinc warehouse stocks total.
- `13890` (CHLDWTL): SHFE lead warehouse stocks weekly total.
- `13896` (CHLDWTS): SHFE lead warehouse stocks Tianjin subtotal.
- `16847` (SNITTOT): SHFE nickel warehouse stocks total.
- `16896` (STNTTOT): SHFE tin warehouse stocks total.

### Steel & Bulk Materials
- `16603` (STRB5CT): Steel rebar inventory China total 35 cities.
- `16639` (STWR5CT): Steel wire rod inventory China total 35 cities.
- `16879` (SHRTTOT): SHFE hot rolled coil warehouse stock total.
- `16486` (IRMIATT): Iron ore inventory import total, Dalian port.
- `16396` (COINIAN): Coke inventory, Tianjin port.

### Industrial Materials
- `5203` (CHFOWHS): SHFE fuel oil warehouse stocks total.
- `5215` (CHNRWHS): SHFE rubber warehouse stocks total (used for NR and TOCOM rubber).
- `2137` (LPYWARE): LME polypropylene warehouse stocks (legacy series).
