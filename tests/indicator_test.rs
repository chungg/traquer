use traquer::indicator;

mod common;

#[test]
fn test_adx() {
    let stats = common::test_data();
    let result = indicator::adx(&stats.high, &stats.low, &stats.close, 14, 14);
    assert_eq!(
        vec![
            31.860139412194698,
            30.900411343329193,
            29.846209753924423,
            28.926644263612655,
            29.10834871990969,
            29.939197830151386,
            29.09419668802638,
            30.29237400866025,
            29.60030469615574,
            28.537858280042748,
            32.473148424041746,
            34.982949940305744,
            33.47283006287883,
            33.22624512306407,
            34.306046041056824,
            32.83133865336625,
            33.02567539621716,
            32.10723238925509,
            37.32862815236502,
            35.23477297887128,
        ],
        result.0
    );
    assert_eq!(
        vec![
            26.730133235533003,
            28.809316234232448,
            28.26198015278067,
            28.174104325301958,
            26.454545579221207,
            25.18695046089198,
            24.47607630765512,
            23.69367885417301,
            23.152365451974088,
            24.818266407121218,
            23.34724730276131,
            22.051913999559698,
            23.91704820756414,
            22.602128450065592,
            21.10029388200336,
            20.193259616585753,
            19.03617008379953,
            18.027067079868925,
            16.20809033889095,
            17.880973133699126,
        ],
        result.1
    );
    assert_eq!(
        vec![
            10.317020581019671,
            11.282545552195204,
            12.179104454001054,
            13.228521011257213,
            14.289690573973955,
            16.086895874937387,
            17.27152240413718,
        ],
        result.2
    );
}

#[test]
fn test_qstick() {
    let stats = common::test_data();
    let result = indicator::qstick(&stats.open, &stats.close, 8);
    assert_eq!(
        vec![
            -0.5343747138977051,
            -0.6200696627298992,
            -0.06449838920875839,
            -0.4301652103785134,
            -0.6456843916008229,
            -0.9266433817811088,
            -0.16961162088834514,
            -0.39858698578859136,
            -0.7411229059931231,
            -0.43420684029499856,
            -0.6110505106591309,
            0.17362782140791555,
            0.20393305515971555,
            0.4163923423269142,
            0.49052737736537766,
            0.4904105553879152,
            0.5614307371441598,
            0.9811129650986694,
            0.8141988711292603,
            0.40437632332406004,
            0.5389598373032619,
            0.6614132406998461,
            0.11443269008642548,
            0.3734473765776365,
            0.6615706225920679,
            -0.2854447322330791,
            -0.6542355716305024,
        ],
        result
    );
}

#[test]
fn test_twiggs() {
    let stats = common::test_data();
    let result = indicator::twiggs(&stats.high, &stats.low, &stats.close, &stats.volume, 16);
    assert_eq!(
        vec![
            -0.18613223690613337,
            -0.19317352339707441,
            -0.17453508236798282,
            -0.18375940487028466,
            -0.17502586620511137,
            -0.17382219844626962,
            -0.17255131614049954,
            -0.1640328604291291,
            -0.1420920809902787,
            -0.1527562191185741,
            -0.16523607317121108,
            -0.16430836437056545,
            -0.15920358586187838,
            -0.16715044705529314,
            -0.1621303949690616,
            -0.11881583358526457,
            -0.14855035203055256,
            -0.15469018900758857,
        ],
        result
    );
}

#[test]
fn test_rsi() {
    let stats = common::test_data();
    let result = indicator::rsi(&stats.close, 16);
    assert_eq!(
        vec![
            46.128106720103226,
            44.633678866135206,
            47.00425158432756,
            46.30758159352003,
            47.45475914349045,
            48.24782813294231,
            48.597941941188665,
            47.8652658468592,
            51.2057484346344,
            52.13878031503008,
            49.934061780601795,
            51.64109828171605,
            53.76628012209768,
            52.337400470511554,
            54.32827223456176,
            55.696698317906126,
            58.06858163571402,
            54.77626809114967,
        ],
        result
    );
}

#[test]
fn test_kvo() {
    let stats = common::test_data();
    let result = indicator::kvo(&stats.high, &stats.low, &stats.close, &stats.volume, 10, 16);
    assert_eq!(
        vec![
            -656322.9297191856,
            -713366.2205723817,
            -454503.008716189,
            -242243.40638757526,
            -150673.24628108775,
            -51176.64414472319,
            -14990.826684893807,
            -172705.0049532547,
            -10141.336752269068,
            101922.08867095085,
            -103711.73979187862,
            -34908.10280425532,
            64722.50363716809,
            -92755.34740538022,
            7627.067027119803,
            344638.3915205905,
            1045791.2002935112,
            491703.71571153356,
        ],
        result
    );
}

#[test]
fn test_macd() {
    let stats = common::test_data();
    let result = indicator::macd(&stats.close, 12, 26);
    assert_eq!(
        vec![
            -2.217893848328025,
            -2.012936386965073,
            -1.7145457437113834,
            -1.316039757031355,
            -1.0733808769373852,
            -0.7410026847983033,
            -0.3820054099132051,
            0.06315775884193187,
            0.24329207900761673,
        ],
        result
    );
}

#[test]
fn test_cmo() {
    let stats = common::test_data();
    let result = indicator::cmo(&stats.close, 16);
    assert_eq!(
        vec![
            -7.743786559793555,
            -40.5065764761353,
            -48.55718616179731,
            -40.50419392251485,
            -21.175095245365554,
            -9.902549550882728,
            -26.391891770751375,
            -18.232928699810707,
            -1.3123299513601174,
            -4.972814726212496,
            -6.534210047973057,
            3.3257784266638946,
            22.112731584577258,
            8.795331918339102,
            25.684922447035984,
            42.304220440657446,
            44.454063824420864,
            43.61402085815939,
        ],
        result
    );
}

#[test]
fn test_cog() {
    let stats = common::test_data();
    let result = indicator::cog(&stats.close, 16);
    assert_eq!(
        vec![
            -8.967512936942779,
            -9.068163820345244,
            -9.040879286087515,
            -8.897050544657091,
            -8.80553442078012,
            -8.772365854633845,
            -8.759057548354765,
            -8.683880858831056,
            -8.646508895454593,
            -8.583784959158981,
            -8.489125991456865,
            -8.429902985473813,
            -8.366876603883997,
            -8.322698920592734,
            -8.267741798560767,
            -8.227747449061596,
            -8.213709147564295,
            -8.179093436115151,
            -8.204185273664265,
        ],
        result
    );
}

#[test]
fn test_shinohara() {
    let stats = common::test_data();
    let results = indicator::shinohara(&stats.high, &stats.low, &stats.close, 26);
    assert_eq!(
        vec![
            119.72458721822203,
            95.38515682324336,
            68.70621869411681,
            85.35865719588207,
            119.54886405652606,
            139.72926160563833,
            150.77427373025645,
            150.9041867287,
        ],
        results.0
    );
    assert_eq!(
        vec![
            130.9311144714237,
            126.54955949420238,
            167.3688494557726,
            146.27994868199906,
            142.35182619352173,
            125.70722906999697,
            118.00938106154118,
            143.57856018147575,
            133.74958218587676,
        ],
        results.1
    );
}

#[test]
fn test_elder_ray() {
    let stats = common::test_data();
    let results = indicator::elder_ray(&stats.high, &stats.low, &stats.close, 16);
    assert_eq!(
        vec![
            -6.323126792907715,
            -5.426287538865033,
            -5.97348870868089,
            -3.8769004733650263,
            -1.5737349488532146,
            -2.4380034539800874,
            -0.690589604456811,
            -0.5922851156483517,
            -1.6737827615957386,
            1.753899378196202,
            4.2190289169688455,
            2.1200258021967784,
            2.9185518732331417,
            4.514192005348761,
            3.396935010966615,
            4.217882038054185,
            4.2369573334537165,
            8.442607464131385,
            4.079358351350422,
        ],
        results.0
    );
    assert_eq!(
        vec![
            -8.743124961853027,
            -8.246287233689252,
            -7.878487487977765,
            -8.656903067359167,
            -5.23373861096259,
            -4.498004827271103,
            -2.9285923815564203,
            -2.1122855734120236,
            -3.4637836771230823,
            -1.7951004081807511,
            1.189026322974705,
            -0.48497374003954974,
            -0.18145041558521768,
            0.5951932870870351,
            0.936935926493959,
            1.0978831061694194,
            1.3069570282779353,
            2.9826083796587284,
            1.229359877229328,
        ],
        results.1
    );
}

#[test]
fn test_elder_force() {
    let stats = common::test_data();
    let result = indicator::elder_force(&stats.close, &stats.volume, 16);
    assert_eq!(
        vec![
            15163924.326658249,
            12880758.096563116,
            12050605.61114628,
            10383051.49386349,
            9364365.879667005,
            8442213.142258454,
            7502743.65895507,
            6497707.801010304,
            6702367.440261993,
            6181683.142965115,
            5074974.977031417,
            4748656.497570231,
            4696379.1776512945,
            3995675.78829447,
            3931504.90271624,
            4298754.639673724,
            6967489.387947403,
            5483494.728818839,
        ],
        result
    );
}
#[test]
fn test_mfi() {
    let stats = common::test_data();
    let result = indicator::mfi(&stats.high, &stats.low, &stats.close, &stats.volume, 16);
    assert_eq!(
        vec![
            57.23427292919309,
            47.90601255639084,
            21.137944754987743,
            30.26117456879261,
            38.51678304832031,
            50.513828792387955,
            40.49727451951461,
            44.32715297759769,
            53.90985418627666,
            53.60555590702833,
            55.07839144768625,
            60.65330461162574,
            70.80167748372554,
            64.11003122868968,
            70.5203113367536,
            82.33336948811267,
            87.34401989374501,
            85.51147984689622,
        ],
        result,
    );
}

#[test]
fn test_ad() {
    let stats = common::test_data();
    let result = indicator::ad(&stats.high, &stats.low, &stats.close, &stats.volume);
    assert_eq!(
        vec![
            -12220014.53999535,
            5594494.456558675,
            -18251656.973511968,
            -27192954.017261956,
            -38906280.775514156,
            -43519171.793909825,
            -36388813.33166392,
            -40306772.321005546,
            -43038979.11826538,
            -41295751.02899514,
            -42530068.87440958,
            -44631518.28622454,
            -47863110.10330983,
            -47175528.73291944,
            -48928369.757053,
            -51464869.93675375,
            -50970834.60616014,
            -52061872.3366441,
            -50169605.03257219,
            -51716854.467872486,
            -50995298.79832875,
            -51206859.811250634,
            -51316121.96369292,
            -50145740.751600124,
            -48579440.279196754,
            -50497562.83272522,
            -51860185.667756006,
            -52138562.67733089,
            -52113820.167047076,
            -52915683.21006013,
            -52969956.24185295,
            -50690564.9792013,
            -55780759.43377636,
            -56496069.02735695,
        ],
        result
    );
}

#[test]
fn test_ad_yahoo() {
    let stats = common::test_data();
    let result = indicator::ad_yahoo(&stats.high, &stats.low, &stats.close, &stats.volume);
    assert_eq!(
        vec![
            336703421.3851929,
            524311079.90493774,
            386249980.45578,
            279963693.60809326,
            236454248.4260559,
            297360960.63041687,
            276863635.63041687,
            264624868.58329773,
            273054631.709671,
            264990289.9585724,
            259319809.13619995,
            246661018.79997253,
            257403780.75141907,
            252853500.13084412,
            245025403.5522461,
            248117583.03375244,
            243874594.40460205,
            253856117.67807007,
            246398220.74928284,
            248672416.60499573,
            250625005.75447083,
            251593806.81037903,
            250553745.6768036,
            258791060.41145325,
            261067611.32469177,
            257009643.54515076,
            259310911.44676208,
            263993610.364151,
            261545710.7322693,
            264995933.9931488,
            277479333.9931488,
            304462333.9931488,
            298815868.7785034,
        ],
        result
    );
}

#[test]
fn test_cmf() {
    let stats = common::test_data();
    let result = indicator::cmf(&stats.high, &stats.low, &stats.close, &stats.volume, 16);
    assert_eq!(
        vec![
            -0.31996402726557827,
            -0.26431568287267065,
            -0.4636131359961498,
            -0.3495321537587002,
            -0.3266108255947919,
            -0.18899633112243233,
            -0.14523458977952866,
            -0.3326493878867275,
            -0.23687295947927223,
            -0.13961905719447967,
            -0.23334715737003828,
            -0.2421809194283885,
            -0.19835662209432817,
            -0.11853261789898396,
            -0.1750773026968712,
            -0.12288304498976765,
            0.021699608427354102,
            -0.1013140453290804,
            -0.09237139625802167,
        ],
        result
    );
}

#[test]
fn test_cvi() {
    let stats = common::test_data();
    let result = indicator::cvi(&stats.high, &stats.low, 16, 2);
    assert_eq!(
        vec![
            -12.903778070881856,
            -7.712487202835517,
            -3.1188656632364697,
            -9.341946193517892,
            -12.034597148362437,
            -12.814471005826123,
            -13.271361266019388,
            -6.439924966331101,
            -2.0112332700986557,
            -4.867518373670898,
            -4.142244814924245,
            0.7501021543131925,
            -1.5573115591711928,
            -4.07753627445564,
            -2.062093533894327,
            6.497958034939444,
            5.52228653707274,
        ],
        result
    );
}

#[test]
fn test_wpr() {
    let stats = common::test_data();
    let result = indicator::wpr(&stats.high, &stats.low, &stats.close, 16);
    assert_eq!(
        vec![
            -99.09142622449394,
            -94.88476784384058,
            -98.7016646516565,
            -83.45322691468988,
            -80.33424822308433,
            -66.49998256138393,
            -60.92856270926339,
            -58.24332819489061,
            -56.88925190670342,
            -31.699065254207774,
            -24.51395669497002,
            -38.50825188642196,
            -26.019074805435594,
            -15.778329765623885,
            -24.403940042613893,
            -12.779540060870623,
            -7.164869527394986,
            -21.111723928174033,
            -32.930944560522704,
        ],
        result
    );
}

#[test]
fn test_vortex() {
    let stats = common::test_data();
    let (vi_pos, vi_neg) = indicator::vortex(&stats.high, &stats.low, &stats.close, 16);
    assert_eq!(
        vec![
            0.8610723090930696,
            0.8159089697456218,
            0.5782198030623583,
            0.7200793857247078,
            0.8358118890986928,
            0.9355813847576413,
            0.9484126915125689,
            0.8489016637989781,
            0.9131108818882286,
            0.9790387062979787,
            0.9343265196480259,
            0.9443134085341751,
            1.0302488811514465,
            1.0364553935724832,
            1.0553301509762798,
            1.1219923299032897,
            1.161732264987062,
            1.1478332770638693,
        ],
        vi_pos
    );
    assert_eq!(
        vec![
            1.029701151945177,
            1.1817046446451998,
            1.3803206728528217,
            1.238892593460365,
            1.1850444607043855,
            1.061167502645104,
            1.111042759028416,
            1.1313347365841422,
            0.9951327158267141,
            0.9280527122256887,
            0.9901265627745084,
            0.9313767804581332,
            0.8402113281909229,
            0.891932290028499,
            0.8280623772231498,
            0.7860652938018367,
            0.6974842970716879,
            0.7557323147066469,
        ],
        vi_neg
    );
}

#[test]
fn test_po() {
    let stats = common::test_data();
    let result = indicator::po(&stats.volume, 10, 16);
    assert_eq!(
        vec![
            -35.378723807894985,
            -37.993168058882375,
            -39.524346350340444,
            -40.41967020222986,
            -40.51097271301698,
            -42.086823387150005,
            -42.30015209438188,
            -43.383528771209576,
            -43.81409605428357,
            -40.648471039972534,
            -37.7876496415686,
            -37.39351505516741,
            -37.136103488993875,
            -34.28067604316157,
            -35.12026222042619,
            -32.44673522414948,
            -18.294669010949182,
            2.308566455542005,
            -0.92080395315155,
        ],
        result
    );
}

#[test]
fn test_vhf() {
    let stats = common::test_data();
    let result = indicator::vhf(&stats.high, &stats.low, &stats.close, 16);
    assert_eq!(
        vec![
            0.5669216159971233,
            0.7107794587594408,
            0.5482664867838217,
            0.4309723520119755,
            0.40721343838432617,
            0.44011310017913846,
            0.5021691775026703,
            0.4751003004570894,
            0.4435695260250405,
            0.4595960007629394,
            0.4405808834558327,
            0.4357521288992697,
            0.4843910183060343,
            0.5122550500601386,
            0.5359587346575249,
            0.5841583342518132,
            0.7716635213608084,
            0.7671760704973449,
        ],
        result
    );
}

#[test]
fn test_ultimate() {
    let stats = common::test_data();
    let result = indicator::ultimate(&stats.high, &stats.low, &stats.close, 6, 12, 24);
    assert_eq!(
        vec![
            52.64489292919164,
            51.59059656807282,
            46.03014177584667,
            46.83402417416914,
            47.63501864800235,
            43.80674742529631,
            38.16680505680669,
            44.10353752395525,
            44.154676988833835,
            42.65072465563253,
        ],
        result
    );
}

#[test]
fn test_pgo() {
    let stats = common::test_data();
    let result = indicator::pgo(&stats.high, &stats.low, &stats.close, 16);
    assert_eq!(
        vec![
            -1.1728195008646218,
            -1.3831996312942632,
            -0.669537387649706,
            -0.6555101010723227,
            -0.37389208904655474,
            -0.18480884776195328,
            -0.01324917609586888,
            -0.11871191726864071,
            0.6407704612678665,
            0.8895192850294852,
            0.4619045740578769,
            0.8450928215439151,
            1.234792786595683,
            0.9512531985339747,
            1.3179726777354788,
            1.5016968220594469,
            1.7570721475194715,
            1.0410621667725752,
        ],
        result
    );
}

#[test]
fn test_si() {
    let stats = common::test_data();
    let result = indicator::si(&stats.open, &stats.high, &stats.low, &stats.close, 0.5);
    assert_eq!(
        vec![
            1863.9824746176116,
            654.8878036623194,
            -1104.4095193965052,
            -1214.2944568105527,
            -487.3407354098372,
            427.5088169169998,
            -223.860381128408,
            -110.7121065452446,
            162.75807618131765,
            -98.12145222473158,
            -101.13846512002326,
            -403.1845902647818,
            283.1891569876117,
            -199.95462669308907,
            -326.86640146468073,
            63.131576090683794,
            -253.67664708736052,
            215.41843387212208,
            2.1629182406584904,
            143.69919207899204,
            115.4626169352983,
            51.354553395932655,
            -18.304465830729768,
            429.0203486740835,
            141.26847935246593,
            -183.79992761728568,
            160.03660663696868,
            235.8667734660081,
            -90.16470772181825,
            162.16835332629032,
            144.83405393167723,
            59.009650555611834,
            -321.26018448483165,
        ],
        result
    );
}

#[test]
fn test_asi() {
    let stats = common::test_data();
    let result = indicator::asi(&stats.open, &stats.high, &stats.low, &stats.close, 0.5);
    assert_eq!(
        vec![
            1863.9824746176116,
            2518.870278279931,
            1414.4607588834258,
            200.16630207287312,
            -287.17443333696406,
            140.33438358003576,
            -83.52599754837223,
            -194.2381040936168,
            -31.480027912299164,
            -129.60148013703076,
            -230.73994525705402,
            -633.9245355218358,
            -350.7353785342241,
            -550.6900052273131,
            -877.5564066919939,
            -814.4248306013101,
            -1068.1014776886707,
            -852.6830438165487,
            -850.5201255758901,
            -706.8209334968981,
            -591.3583165615997,
            -540.0037631656671,
            -558.3082289963969,
            -129.2878803223134,
            11.980599030152518,
            -171.81932858713316,
            -11.782721950164472,
            224.08405151584364,
            133.9193437940254,
            296.08769712031574,
            440.92175105199294,
            499.93140160760476,
            178.6712171227731,
        ],
        result
    );
}

#[test]
fn test_ulcer() {
    let stats = common::test_data();
    let result = indicator::ulcer(&stats.close, 8);
    assert_eq!(
        vec![
            20.73223668909602,
            19.094435269396094,
            16.649343658592862,
            14.662110635179326,
            13.083755073024895,
            12.845054285539643,
            11.60084753454864,
            10.833635700777663,
            10.094706264836718,
            8.405227208060785,
            6.91887557627602,
            4.390517821635424,
            3.822504042165251,
            2.511303218451697,
            1.5486160074003668,
            1.7365186635053362,
            1.7365186635053362,
            1.6390653783421978,
            1.6390653783421978,
            2.187009055695484,
        ],
        result
    );
}

#[test]
fn test_tr() {
    let stats = common::test_data();
    let result = indicator::tr(&stats.high, &stats.low, &stats.close);
    assert_eq!(
        vec![
            15.939998626708984,
            15.10000228881836,
            9.490001678466797,
            8.650001525878906,
            4.924999237060547,
            7.349998474121094,
            4.69000244140625,
            3.3300018310546875,
            3.6100006103515625,
            4.1399993896484375,
            2.5900001525878906,
            3.279998779296875,
            4.340000152587891,
            2.3699989318847656,
            2.5900001525878906,
            2.8199996948242188,
            2.4399986267089844,
            4.780002593994141,
            3.660003662109375,
            2.0600013732910156,
            2.2380027770996094,
            1.5200004577636719,
            2.3000030517578125,
            3.7490005493164063,
            3.450000762939453,
            2.604999542236328,
            3.2600021362304688,
            3.918998718261726,
            2.4599990844726563,
            3.229999542236328,
            2.9300003051757813,
            5.759998321533203,
            3.1500015258789063,
        ],
        result
    );
}

#[test]
fn test_hlc3() {
    let stats = common::test_data();
    let result = indicator::hlc3(&stats.high, &stats.low, &stats.close, 16);
    assert_eq!(
        vec![
            48.82131250699361,
            48.41006247202555,
            47.38204161326091,
            45.67329160372416,
            44.58475001653036,
            43.98891671498617,
            43.760937531789146,
            43.422812620798744,
            43.03031253814698,
            42.92550015449524,
            42.935500065485634,
            42.83081253369649,
            42.84727080663045,
            43.15249999364217,
            43.33354179064433,
            43.67937509218853,
            44.2075002193451,
            44.90875029563905,
            45.53729192415874,
        ],
        result
    );
}

#[test]
fn test_trix() {
    let stats = common::test_data();
    let result = indicator::trix(&stats.close, 7);
    assert_eq!(
        vec![
            -1.7609812348121436,
            -1.58700358125189,
            -1.3595873824994853,
            -1.1099628496419054,
            -0.8955729429209658,
            -0.6091528694171965,
            -0.2949587326281726,
            -0.07920030554104754,
            0.1135712345224751,
            0.32952700679764474,
            0.4769166955410566,
            0.6219613185170387,
            0.7718438325710452,
            0.9560958810856369,
            1.0335513961870586,
        ],
        result
    );
}

#[test]
fn test_tii_even() {
    let stats = common::test_data();
    let result = indicator::tii(&stats.close, 16);
    assert_eq!(
        vec![
            0.0,
            0.0,
            12.365592243625398,
            36.838871014658466,
            53.81832516603384,
            77.25510120423154,
            91.83894521268712,
            97.2706485470137,
            98.02691957272636,
            100.0,
            100.0,
            100.0,
        ],
        result
    );
}

#[test]
fn test_tii_odd() {
    let stats = common::test_data();
    let result = indicator::tii(&stats.close, 15);
    assert_eq!(
        vec![
            0.0,
            0.6686354433953655,
            0.9316341618307428,
            17.111743788068527,
            44.82345282880815,
            61.21209452789354,
            83.24057181663895,
            96.05752982533481,
            98.58952713255982,
            98.83515680783523,
            100.0,
            100.0,
            100.0,
        ],
        result
    );
}

#[test]
fn test_tvi() {
    let stats = common::test_data();
    let result = indicator::tvi(&stats.close, &stats.volume, 0.5);
    assert_eq!(
        vec![
            24398800.0, 59729800.0, 40971500.0, 28363400.0, 15375500.0, 24818400.0, 19995500.0,
            15377100.0, 18304100.0, 15642600.0, 13365300.0, 9015200.0, 13278200.0, 11264800.0,
            7816300.0, 9515300.0, 7361500.0, 9645600.0, 7117500.0, 8603900.0, 10560400.0,
            11944400.0, 10458600.0, 13222800.0, 15901100.0, 14148200.0, 15746300.0, 18111300.0,
            16923000.0, 19039700.0, 25281400.0, 38772900.0, 36090498.0,
        ],
        result
    );
}

#[test]
fn test_supertrend() {
    let stats = common::test_data();
    let result = indicator::supertrend(&stats.high, &stats.low, &stats.close, 16, 3.0);
    assert_eq!(
        vec![
            22.87718629837036,
            22.87718629837036,
            22.87718629837036,
            25.36115028045606,
            25.5548271386142,
            27.535275836318306,
            28.482134516844127,
            28.482134516844127,
            30.372852510605856,
            33.54470377638434,
            33.54470377638434,
            33.54470377638434,
            34.55477737989572,
            34.70432339242238,
            35.73655158775987,
            36.52801923545023,
            39.78407957956028,
            39.78407957956028,
        ],
        result
    );
}
