# Urban Heat Island & Environmental Domain Research

Sources that ground **SPARC's domain templates**, **physics priors**, **intervention constraints**, and the **Providence UHI study** — the primary validation dataset cited throughout the codebase and published in *Urban Climate* (2025).

---

## SPARC Publication

> **Wire, K. (2025).** "SPARC: A spatially-weighted causal inference pipeline for urban heat island analysis." *Urban Climate*, 52, 102671. https://doi.org/10.1016/j.uclim.2025.102671

The published validation of SPARC v1 on the Providence, RI urban heat island dataset. The current pipeline (v2+) has since reached 94.4% R² on the same dataset through the SharedTrunk + CityHead architecture, Laplacian eigenmaps, and staged 10-term PDE curriculum — roughly +9 pp over the published results.

---

## Urban Heat Island — Foundational Theory

> **Oke, T.R. (1982).** "The energetic basis of the urban heat island." *Quarterly Journal of the Royal Meteorological Society*, 108(455), 1–24. https://doi.org/10.1002/qj.49710845502

> **Oke, T.R. (1987).** *Boundary Layer Climates*, 2nd ed. Methuen. ISBN: 978-0415043199.

> **Arnfield, A.J. (2003).** "Two decades of urban climate research: A review of turbulence, exchanges of energy and water, and the urban heat island." *International Journal of Climatology*, 23(1), 1–26. https://doi.org/10.1002/joc.859

These texts establish the thermodynamic and energy-balance foundations for the UHI domain template. Oke (1982) in particular is the theoretical basis for SPARC's surface energy balance PDE term (Q* − Q_H − Q_E) and the sign expectations encoded in the DAG (canopy → cooling, impervious → warming, albedo → cooling).

---

## Land Surface Temperature & Remote Sensing

> **Voogt, J.A., & Oke, T.R. (2003).** "Thermal remote sensing of urban climates." *Remote Sensing of Environment*, 86(3), 370–384. https://doi.org/10.1016/S0034-4257(03)00079-8

> **Weng, Q. (2009).** "Thermal infrared remote sensing for urban climate and environmental studies: Methods, applications, and trends." *ISPRS Journal of Photogrammetry and Remote Sensing*, 64(4), 335–344. https://doi.org/10.1016/j.isprsjprs.2009.03.007

Remote sensing provides the spatially-continuous temperature fields and predictor layers (NDVI, albedo, land cover) that are SPARC's primary inputs. Voogt & Oke (2003) covers the distinction between surface temperature (from thermal IR sensors) and ambient air temperature (which SPARC predicts), and the correction procedures needed to convert between them.

---

## NDVI and Vegetation Cooling

> **Tucker, C.J. (1979).** "Red and photographic infrared linear combinations for monitoring vegetation." *Remote Sensing of Environment*, 8(2), 127–150. https://doi.org/10.1016/0034-4257(79)90013-0

> **Buyantuyev, A., & Wu, J. (2010).** "Urban heat islands and landscape heterogeneity: Linking spatiotemporal variations in surface temperatures to land-cover and socioeconomic patterns." *Landscape Ecology*, 25(1), 17–33. https://doi.org/10.1007/s10980-009-9402-4

NDVI is used as both a predictor and a causal mediator in the `uhi` template DAG (Canopy → NDVI → Temperature). Tucker (1979) establishes the NDVI formula; Buyantuyev & Wu (2010) demonstrates vegetation as a primary UHI moderator at the neighborhood scale — consistent with the large NDVI → AAT_z coefficient (−4.131) estimated in the Providence study.

---

## Impervious Surfaces and Urban Heat

> **Xu, H. (2008).** "A new index for delineating built-up land features in satellite imagery." *International Journal of Remote Sensing*, 29(14), 4269–4276. https://doi.org/10.1080/01430160801996803

> **Oke, T.R., Mills, G., Christen, A., & Voogt, J.A. (2017).** *Urban Climates.* Cambridge University Press. ISBN: 978-1107429010.

The positive causal coefficient for `Pct_Impervious → AAT_z` (+0.022 per pp in the Providence study) is consistent with the extensive literature documenting impervious surfaces' role in UHI through reduced evapotranspiration, increased solar absorption, and anthropogenic heat storage. Oke et al. (2017) is the definitive modern textbook for urban climate physics.

---

## Albedo and Surface Reflectance

> **Taha, H. (1997).** "Urban climates and heat islands: Albedo, evapotranspiration, and anthropogenic heat." *Energy and Buildings*, 25(2), 99–103. https://doi.org/10.1016/S0378-7788(96)00999-1

> **Santamouris, M. (2014).** "Cooling the cities – A review of reflective and green roof mitigation technologies to fight heat island and improve comfort in urban environments." *Solar Energy*, 103, 682–703. https://doi.org/10.1016/j.solener.2012.07.003

Albedo is a key intervention variable in Stage 4 scenarios (cool roofs, reflective pavements). The large negative coefficient (Albedo → AAT_z: −2.759) validates the physical prior that higher surface reflectance reduces sensible heat generation. Taha (1997) and Santamouris (2014) establish the plausible effect-size ranges used in SPARC's physics priors and intervention caps.

---

## Spatial Heterogeneity of UHI Effects

> **Zhao, L., Lee, X., Smith, R.B., & Oleson, K. (2014).** "Strong contributions of local background climate to urban heat islands." *Nature*, 511, 216–219. https://doi.org/10.1038/nature13462

> **Imhoff, M.L., Zhang, P., Wolfe, R.E., & Bounoua, L. (2010).** "Remote sensing of the urban heat island effect across biomes in the continental USA." *Remote Sensing of Environment*, 114(3), 504–513. https://doi.org/10.1016/j.rse.2009.10.008

The strong spatial non-stationarity of UHI (different causal mechanisms in downtown cores vs. residential neighborhoods vs. riparian zones) motivates SPARC's geographically-weighted approach over global regression. GWR's spatially-varying coefficients capture this heterogeneity; the CATE maps from Stage 3 quantify where each intervention is most effective.

---

## ForceSMIP — Climate Forcing Attribution

> **Zelinka, M.D., Myers, T.A., McCoy, D.T., Po-Chedley, S., Caldwell, P.M., Ceppi, P., Klein, S.A., & Taylor, K.E. (2020).** "Causes of Higher Climate Sensitivity in CMIP6 Models." *Geophysical Research Letters*, 47(1), e2019GL085782. https://doi.org/10.1029/2019GL085782

> **Santer, B.D., Mears, C., Doutriaux, C., Caldwell, P., Gleckler, P.J., Wigley, T.M.L., Solomon, S., Gillett, N.P., Ivanova, D., Karl, T.R., Lanzante, J.R., Meehl, G.A., Stott, P.A., Taylor, K.E., Thorne, P.W., & Wehner, M.F. (2011).** "Separating signal and noise in atmospheric temperature changes: The importance of timescale." *Journal of Geophysical Research: Atmospheres*, 116, D22105. https://doi.org/10.1029/2011JD016263

SPARC has been applied to ForceSMIP (Forced Response Model Intercomparison Project) Tier 1 at global scale — attributing climate model output to individual forcing agents (GHG, aerosols, solar, volcanic). The `forcesmip` domain template adapts SPARC's causal inference framework from local UHI to global climate attribution, with the DAG encoding GHG forcing → temperature response → regional amplification pathways.

---

## Groundwater / Hydrogeology Domain

> **Freeze, R.A., & Cherry, J.A. (1979).** *Groundwater.* Prentice Hall. ISBN: 978-0133653120.

> **Dagan, G. (1989).** *Flow and Transport in Porous Formations.* Springer. https://doi.org/10.1007/978-3-642-75015-1

The `groundwater` template models groundwater level and contaminant transport. The physics priors encode Darcy's law (flow proportional to hydraulic gradient, bounded by hydraulic conductivity) and the advection-dispersion equation for contaminant migration. SPARC's Tier 2 PDE solver applies these as a forward model under intervention scenarios (e.g., pump-and-treat remediation).

---

## Air Quality Domain

> **Seinfeld, J.H., & Pandis, S.N. (2016).** *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change*, 3rd ed. Wiley. ISBN: 978-1118947401.

> **Pope, C.A., & Dockery, D.W. (2006).** "Health Effects of Fine Particulate Air Pollution: Lines that Connect." *Journal of the Air & Waste Management Association*, 56(6), 709–742. https://doi.org/10.1080/10473289.2006.10464485

The `air_quality` template models PM2.5 and NOx concentration fields. The physics prior encodes Gaussian dispersion from point sources (the Briggs plume-rise equations) and the downwind decay relationship. The causal DAG identifies emission sources, meteorological drivers (wind speed, boundary layer height), and receptors.

---

## Wildfire Domain

> **Rothermel, R.C. (1972).** *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* USDA Forest Service Research Paper INT-115.

> **Scott, J.H., & Burgan, R.E. (2005).** *Standard Fire Behavior Fuel Models: A Comprehensive Set for Use with Rothermel's Surface Fire Spread Model.* USDA Forest Service General Technical Report RMRS-GTR-153.

The `wildfire` template's physics priors encode Rothermel's fire spread rate equation (a function of wind, slope, and fuel moisture) as constraint bounds on intervention effects. The causal DAG models fuel load, ignition probability, weather variables, and suppression capacity as drivers of burn severity.

---

## Seismic Hazard Domain

> **Kramer, S.L. (1996).** *Geotechnical Earthquake Engineering.* Prentice Hall. ISBN: 978-0133749434.

> **Atkinson, G.M., & Boore, D.M. (2003).** "Empirical Ground-Motion Relations for Subduction-Zone Earthquakes and Their Application to Cascadia and Other Regions." *Bulletin of the Seismological Society of America*, 93(4), 1703–1729. https://doi.org/10.1785/0120020156

The `seismic` template models ground motion intensity (PGA, SA) as a function of source magnitude, distance, and site amplification (Vs30). The physics priors encode ground motion prediction equations (GMPEs) as bounds on the achievable attenuation. SPARC's CATE estimation identifies where soil conditions amplify or attenuate shaking spatially.
