use traquer::smooth;

mod common;

#[test]
fn test_ewma_even() {
    let stats = common::test_data();
    let results = smooth::ewma(&stats.close, 16);
    assert_eq!(
        vec![
            48.42312526702881,
            47.56628692851347,
            46.57848825091722,
            46.00690154148026,
            45.4037367799079,
            45.008003148804306,
            44.75059097774783,
            44.56228633635148,
            44.313782151244176,
            44.44510193405966,
            44.66097215114639,
            44.634975265918456,
            44.78144888970631,
            45.12480793361609,
            45.30306575197284,
            45.65211689383058,
            46.09304449760097,
            46.717392383280725,
            47.02064012277067,
        ],
        results
    );
}

#[test]
fn test_ewma_odd() {
    let stats = common::test_data();
    let results = smooth::ewma(&stats.close, 15);
    assert_eq!(
        vec![
            48.984666951497395,
            47.86158358256022,
            47.02138555844625,
            46.03996213475863,
            45.49996702050169,
            44.922471276453386,
            44.56216248133763,
            44.344392133023455,
            44.195093307130385,
            43.97695673910652,
            44.158587184865176,
            44.42376363416914,
            44.425793008236624,
            44.60756901572145,
            44.994122984123706,
            45.1998575348143,
            45.58362540018297,
            46.060672415894956,
            46.72808855464295,
            47.048952256430745,
        ],
        results
    );
}

#[test]
fn test_sma_even() {
    let stats = common::test_data();
    let results = smooth::sma(&stats.close, 16);
    assert_eq!(
        vec![
            48.42312526702881,
            48.119375228881836,
            46.830000162124634,
            45.368125200271606,
            44.31375026702881,
            43.858750343322754,
            43.66187524795532,
            43.20562529563904,
            42.92187523841858,
            42.90000033378601,
            42.82000017166138,
            42.71312499046326,
            42.76812505722046,
            43.11875009536743,
            43.250625133514404,
            43.625625133514404,
            44.213125228881836,
            44.85437536239624,
            45.48718786239624,
        ],
        results
    );
}

#[test]
fn test_sma_odd() {
    let stats = common::test_data();
    let results = smooth::sma(&stats.close, 15);
    assert_eq!(
        vec![
            48.984666951497395,
            48.5846669514974,
            47.340666961669925,
            45.61133346557617,
            44.542666880289715,
            43.98000030517578,
            43.71800028483073,
            43.20933354695638,
            42.95333353678385,
            42.73133366902669,
            42.589333597819014,
            42.598000081380206,
            42.56066665649414,
            42.813333384195964,
            43.02466684977214,
            43.316000111897786,
            43.8673334757487,
            44.41800028483073,
            45.233333841959634,
            45.73833363850911,
        ],
        results
    );
}

#[test]
fn test_vidya_even() {
    let stats = common::test_data();
    let results = smooth::vidya(&stats.close, 16);
    assert_eq!(
        vec![
            1.658577313274186,
            4.595364682787511,
            8.76102481381672,
            12.6254397400891,
            16.467253504656913,
            19.372387964960026,
            21.950490254244613,
            25.159530096287025,
            27.81046849580971,
            31.240918773298006,
            33.780320050418,
        ],
        results
    );
}

#[test]
fn test_vidya_odd() {
    let stats = common::test_data();
    let results = smooth::vidya(&stats.close, 15);
    assert_eq!(
        vec![
            1.7622383953538225,
            4.87518528044527,
            9.271488225125937,
            13.318684889779746,
            17.315517637570544,
            20.318389791980902,
            22.962590578805536,
            26.24108118694257,
            28.93203238640919,
            32.40359132105506,
            34.92794816477713,
        ],
        results
    );
}

#[test]
fn test_wma_even() {
    let stats = common::test_data();
    let results = smooth::wma(&stats.close, 16);
    assert_eq!(
        vec![
            45.75977967767154,
            44.902941339156214,
            43.85007344975192,
            43.24889710370232,
            42.72088249991922,
            42.45338257621316,
            42.33117661756627,
            42.27095617967493,
            42.18205917582793,
            42.47713271309348,
            42.87477958903593,
            43.06536764257094,
            43.43794129876529,
            44.018161970026355,
            44.43242659288294,
            45.022941336912275,
            45.70227973601397,
            46.547794594484216,
            47.0702208070194,
        ],
        results
    );
}

#[test]
fn test_wma_odd() {
    let stats = common::test_data();
    let results = smooth::wma(&stats.close, 15);
    assert_eq!(
        vec![
            46.527750301361095,
            45.404666932423915,
            44.474083487192786,
            43.45274988810222,
            42.966333357493085,
            42.50850013097127,
            42.26600020726522,
            42.1537501335144,
            42.146333630879724,
            42.083417034149164,
            42.42075036366781,
            42.88208351135254,
            43.11233332951864,
            43.527250130971275,
            44.13808355331422,
            44.59000012079874,
            45.209250164031985,
            45.900833670298255,
            46.773583825429284,
            47.281291866302496,
        ],
        results
    );
}

#[test]
fn test_vma_even() {
    let stats = common::test_data();
    let results = smooth::vma(&stats.close, 16);
    assert_eq!(
        vec![
            12.559081048015681,
            13.753189597123288,
            14.814783120008226,
            15.736656145691265,
            16.25887676661401,
            16.40690598958511,
            16.733357912803474,
            16.77974534035337,
            18.24982174953651,
            19.643736701500703,
            20.921681873074967,
            22.040261760843737,
            23.770359533698354,
            24.81921577730553,
            26.00783926433259,
            27.286658108644293,
            29.00800914486834,
            29.672845402427125,
        ],
        results
    );
}

#[test]
fn test_vma_odd() {
    let stats = common::test_data();
    let results = smooth::vma(&stats.close, 15);
    assert_eq!(
        vec![
            11.935543349426927,
            13.224364249350236,
            14.461385584636243,
            15.56076610496571,
            16.512223321889756,
            17.05072242651991,
            17.20331458093176,
            17.539840412730257,
            17.587581408971978,
            19.105495942260415,
            20.54131919402062,
            21.849985282890504,
            22.99427149901924,
            24.76415641955816,
            25.830139663811828,
            27.038610086893506,
            28.337482288822297,
            30.086715436149444,
            30.75554364251794,
        ],
        results
    );
}

#[test]
fn test_vma_small_window() {
    let stats = common::test_data();
    let results = smooth::vma(&stats.close, 7);
    assert_eq!(
        vec![
            0.37715362566468247,
            4.622867058251505,
            10.844013599747367,
            15.18496527747132,
            16.76859215753913,
            17.88654500033175,
            20.944850749647326,
            22.727313645613926,
            24.29520376753355,
            25.700739774677984,
            26.841565532346944,
            27.48277637144966,
            27.664414368141788,
            28.06610727870126,
            28.121241520722833,
            30.008521258165995,
            31.72799821827373,
            33.120189272434246,
            34.33540917813004,
            36.25025159078928,
            37.262811159876335,
            38.448367775391816,
            39.72062962752442,
            41.49233065069245,
            42.03570500577861,
        ],
        results
    );
}

#[test]
fn test_hull_even() {
    let stats = common::test_data();
    let results = smooth::hull(&stats.close, 16);
    assert_eq!(
        vec![
            39.52407458124597,
            39.376835387049155,
            39.55042884988722,
            40.025559802148855,
            40.71055601219726,
            41.406150204527606,
            42.38044363688798,
            43.5624974082498,
            44.48106479582444,
            45.30848860242009,
            46.159912574057486,
            46.82134468758029,
            47.52666904910717,
            48.29279342539171,
            49.237479216756384,
            49.964879527123145,
        ],
        results
    );
}

#[test]
fn test_hull_odd() {
    let stats = common::test_data();
    let results = smooth::hull(&stats.close, 15);
    assert_eq!(
        vec![
            40.843907204380734,
            40.046115124667125,
            39.721994749705,
            39.586976522869534,
            39.84200009593256,
            40.35702341574209,
            41.10171350373162,
            41.776297060648595,
            42.7662277398286,
            43.975824703993624,
            44.831069552456896,
            45.513768434524536,
            46.301856447149206,
            46.939060115814215,
            47.59569436179268,
            48.33557422779225,
            49.34222279301396,
            50.0292250280027,
        ],
        results
    );
}

#[test]
fn test_dema_even() {
    let stats = common::test_data();
    let results = smooth::dema(&stats.close, 14);
    assert_eq!(
        vec![
            43.15529547948386,
            43.705869121555104,
            44.611040038028165,
            45.082016264158355,
            45.869158373191325,
            46.78431590465494,
            48.0159797504427,
            48.477344590672004,
        ],
        results
    );
}

#[test]
fn test_dema_odd() {
    let stats = common::test_data();
    let results = smooth::dema(&stats.close, 13);
    assert_eq!(
        vec![
            41.878167223746416,
            42.82091685522181,
            43.095979440396235,
            43.707536664699774,
            44.68403708329277,
            45.1817913872971,
            46.00969703692985,
            46.9638790918913,
            48.245239835005584,
            48.692560341186216,
        ],
        results
    );
}

#[test]
fn test_tema_even() {
    let stats = common::test_data();
    let results = smooth::tema(&stats.close, 8);
    assert_eq!(
        vec![
            41.85245710750277,
            42.65215910237417,
            42.76254104206679,
            44.39142477599313,
            45.78886401297522,
            45.58937539753736,
            46.155610612438615,
            47.3905754872981,
            47.49247104352258,
            48.329598253168975,
            49.34731815107276,
            50.92541420046855,
            50.65205563565898,
        ],
        results
    );
}

#[test]
fn test_tema_odd() {
    let stats = common::test_data();
    let results = smooth::tema(&stats.close, 7);
    assert_eq!(
        vec![
            40.30171911199053,
            40.4845463271717,
            41.32284194234779,
            42.27944706651046,
            43.00666092250643,
            42.981805405622296,
            44.653369906246084,
            46.02578255948713,
            45.63917998399449,
            46.16091377840394,
            47.42462810599313,
            47.42758713990865,
            48.27542907351973,
            49.312771023085006,
            50.948867388256396,
            50.52826712062671,
        ],
        results
    );
}

#[test]
fn test_wilder_even() {
    let stats = common::test_data();
    let results = smooth::wilder(&stats.close, 16);
    assert_eq!(
        vec![
            48.42312526702881,
            47.967929899692535,
            47.418059166520834,
            47.06193054490723,
            46.67555995260773,
            46.385837512790204,
            46.16297264916733,
            45.9746619539618,
            45.7543706295229,
            45.73409748425121,
            45.768216315191566,
            45.685202709661404,
            45.697377607064766,
            45.82254155430694,
            45.873632669015784,
            46.023405655812525,
            46.23444289769167,
            46.55729031195337,
            46.72839705301537,
        ],
        results
    );
}

#[test]
fn test_wilder_odd() {
    let stats = common::test_data();
    let results = smooth::wilder(&stats.close, 15);
    assert_eq!(
        vec![
            48.984666951497395,
            48.385689154730905,
            47.90264317039207,
            47.32046683696228,
            46.947102462545004,
            46.542629036249686,
            46.242453828201526,
            46.01429021930971,
            45.823337639747656,
            45.59844851462711,
            45.587218633997026,
            45.63340397701701,
            45.55384362032981,
            45.57558745018217,
            45.71721500436599,
            45.77873396338482,
            45.944818396343415,
            46.17516393831245,
            46.523486444150215,
            46.70825389246989,
        ],
        results
    );
}

#[test]
fn test_lrf() {
    let stats = common::test_data();
    let result = smooth::lrf(&stats.close, 16);
    assert_eq!(
        vec![
            40.43308849895702,
            38.47007355970494,
            37.890220025006464,
            39.01044091056375,
            39.53514696570004,
            39.64264704199398,
            39.66977935678818,
            40.40161794774673,
            40.70242705064662,
            41.63139747170841,
            42.98433842378505,
            43.769852946786315,
            44.77757378185497,
            45.816985719344196,
            46.79602951162002,
            47.817573743707996,
            48.68058875027825,
            49.934633058660175,
            50.23628669626572,
        ],
        result
    );
}

#[test]
fn test_trima_even() {
    let stats = common::test_data();
    let result = smooth::trima(&stats.close, 16);
    assert_eq!(
        vec![
            48.20791710747613,
            47.198889308505585,
            46.07638925976224,
            45.06236139933268,
            44.254444652133515,
            43.622916804419624,
            43.068055629730225,
            42.5454167260064,
            42.1704167260064,
            41.95208342870077,
            41.89694457583957,
            41.99611128701104,
            42.22625022464328,
            42.59222247865465,
            43.011944717831085,
            43.52652804056803,
            44.12541691462199,
            44.77319468392266,
            45.403264098697235,
        ],
        result
    );
}

#[test]
fn test_trima_odd() {
    let stats = common::test_data();
    let result = smooth::trima(&stats.close, 15);
    assert_eq!(
        vec![
            48.712344229221344,
            47.64968794584274,
            46.51796919107437,
            45.4464066028595,
            44.60187524557114,
            43.892187654972076,
            43.29468756914139,
            42.69296878576279,
            42.23281252384186,
            41.9201563000679,
            41.74703133106232,
            41.81609392166138,
            41.99687522649765,
            42.32015651464462,
            42.73265653848648,
            43.23156279325485,
            43.79671901464462,
            44.43218773603439,
            45.093906462192535,
            45.69164079427719,
        ],
        result
    );
}
