************
*Stata Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
*by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter
*17.8.2020
*
*In this code, the data is prepared to be used in the Python code
*This includes clustering the data and running the Poisson estimations
*
*if not installed: ssc install parmest
*also, ensure there is a subfoder "intermediate" in the chosen path
************
clear all

*Set path
cd "D:/RL_Project/"

*Import data from Kitagawa/Tetenov (supplement to their 2018 Econometrica paper from the Econometric Society)
import delimited "jtpa_kt.tab"

*Rename variables
rename d allocated_trmt
rename edu education
rename prevearn prev_earnings

save "jtpa_kt.dta",replace

*For additional variables, use original data from the JTPA National Evaluation (from Upjohn Institute)
clear all
use "expbif.dta"

*Arrival date
gen helper=substr(ra_dt,1,2)+"/"+substr(ra_dt,3,2)+"/"+substr(ra_dt,5,2)
gen date=date(helper,"YMD",1989)
format date %td
drop helper

*Age
destring age, replace

*Keep desired additional information and merge with Kitagawa/Tetenov for identical sample
keep recid date age
destring recid, replace
merge 1:1 recid using "jtpa_kt.dta"
keep if _merge==3
drop _merge

*save
save "the_table_ecma.dta", replace
export delimited using "the_table_ecma.csv", delimiter(tab) replace

*now cluster
foreach var in age prev_earnings education {
egen mean_`var'=mean(`var')
egen sd_`var'=sd(`var')
gen std_`var'=(`var'-mean_`var')/sd_`var'
}

cluster kmedians std_age std_prev_earnings std_education, k(4) name(clstrs) s(kr(21072018))

*relabel clusters such that the first cluster the high-previous-earning one,
*the second cluster the old one, and the third cluster the poorly educated one.
*(can also be skipped of course)
gen clstrs9 = 1
replace clstrs9 = 2 if clstrs==3
replace clstrs9 = 3 if clstrs==1
replace clstrs9 = 4 if clstrs==2
drop clstrs

*save again
save "the_table_ecma_withClusters.dta", replace
export delimited using "the_table_ecma_withClusters.csv", delimiter(tab) replace

*Poisson
gen Year=year(date)

tostring date, gen(datestring)
sort date
encode datestring, gen(dayofyear)

* workyear should start at 1 and go to 252 - and not every day observed
* (especiall yin 1987 and 1989)
replace dayofyear=dayofyear-25 if Year==1988
replace dayofyear=dayofyear+227 if Year==1987
replace dayofyear=dayofyear-277 if Year==1989

foreach c of num 1/4 {
preserve
keep if clstrs9==`c'
gen helper=1
bysort date: egen num_`c'=sum(helper)
bysort date: gen dropper=_n
keep if dropper==1
drop dropper
gen dayofyear_sin=sin(2*_pi*dayofyear/252)
gen dayofyear_cos=cos(2*_pi*dayofyear/252)
poisson num_`c' dayofyear_*
parmest, saving ("intermediate\sincos_coefficients_clusters9_trend_`c'.dta", replace)
restore
}

clear all
use "intermediate\sincos_coefficients_clusters9_trend_1.dta"

foreach c of num 2/4 {
append using "intermediate\sincos_coefficients_clusters9_trend_`c'.dta"
}

destring eq, ignore("num_") gen(clstrs9)

drop eq z

gen Instructions="poisson_mean=exp(c+b_dayofyear_sin*sin(2*pi*dayofyear/250)+b_dayofyear_cos*cos(2*pi*dayofyear/250))"

save "intermediate\sincos_poisson_means_clusters9_trend.dta", replace
export delimited using "sincos_poisson_means_clusters9.csv", delimiter(tab) replace
