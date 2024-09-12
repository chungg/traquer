use traquer::volume;

mod common;

#[test]
fn test_twiggs() {
    let stats = common::test_data();
    let result = volume::twiggs(&stats.high, &stats.low, &stats.close, &stats.volume, 16)
        .collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
        result[16..]
    );
}

#[test]
fn test_kvo() {
    let stats = common::test_data();
    let result = volume::kvo(
        &stats.high,
        &stats.low,
        &stats.close,
        &stats.volume,
        10,
        16,
        Some(false),
    )
    .collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            47653124.52312139,
            38871537.60795313,
            56551756.57617685,
            80719127.50187302,
            91381485.15827936,
            102487748.59386584,
            103133145.86686252,
            76544074.74621749,
            76362373.25709169,
            84632468.62962747,
            54766782.00020293,
            53496835.81130403,
            60063221.578486785,
            36156452.68826012,
            39470196.1311378,
            81568025.19964269,
            163166676.59742773,
            90486708.71792603
        ],
        result[16..]
    );
}

#[test]
fn test_kvo_alt() {
    let stats = common::test_data();
    let result = volume::kvo(
        &stats.high,
        &stats.low,
        &stats.close,
        &stats.volume,
        10,
        16,
        None,
    )
    .collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
        result[16..]
    );
}

#[test]
fn test_elder_force() {
    let stats = common::test_data();
    let result = volume::elder_force(&stats.close, &stats.volume, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
        result[16..]
    );
}
#[test]
fn test_mfi() {
    let stats = common::test_data();
    let result =
        volume::mfi(&stats.high, &stats.low, &stats.close, &stats.volume, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
        result[16..]
    );
}

#[test]
fn test_ad() {
    let stats = common::test_data();
    let result = volume::ad(
        &stats.high,
        &stats.low,
        &stats.close,
        &stats.volume,
        Some(false),
    )
    .collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
    assert_eq!(
        volume::ad(&stats.high, &stats.low, &stats.close, &stats.volume, None)
            .collect::<Vec<f64>>(),
        volume::ad(
            &stats.high,
            &stats.low,
            &stats.close,
            &stats.volume,
            Some(false)
        )
        .collect::<Vec<f64>>()
    );
}

#[test]
fn test_ad_yahoo() {
    let stats = common::test_data();
    let result = volume::ad(
        &stats.high,
        &stats.low,
        &stats.close,
        &stats.volume,
        Some(true),
    )
    .collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
        result[1..]
    );
}

#[test]
fn test_cmf() {
    let stats = common::test_data();
    let result =
        volume::cmf(&stats.high, &stats.low, &stats.close, &stats.volume, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
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
        result[15..]
    );
}

#[test]
fn test_tvi() {
    let stats = common::test_data();
    let result = volume::tvi(&stats.close, &stats.volume, 0.5).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            24398800.0, 59729800.0, 40971500.0, 28363400.0, 15375500.0, 24818400.0, 19995500.0,
            15377100.0, 18304100.0, 15642600.0, 13365300.0, 9015200.0, 13278200.0, 11264800.0,
            7816300.0, 9515300.0, 7361500.0, 9645600.0, 7117500.0, 8603900.0, 10560400.0,
            11944400.0, 10458600.0, 13222800.0, 15901100.0, 14148200.0, 15746300.0, 18111300.0,
            16923000.0, 19039700.0, 25281400.0, 38772900.0, 36090498.0,
        ],
        result[1..]
    );
}

#[test]
fn test_ease() {
    let stats = common::test_data();
    let result = volume::ease(&stats.high, &stats.low, &stats.volume, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            -30.071767986066288,
            -59.754365294318205,
            -89.72212063240552,
            -43.87382059458527,
            -24.838365108537467,
            -2.21638424845535,
            -4.444156166439923,
            -22.889957868500268,
            8.210345571673388,
            31.16013780318997,
            7.3202867183723725,
            22.28059835026842,
            50.675019381222306,
            41.46788293541263,
            55.380575515295135,
            66.30713683458652,
            76.98413524618097,
            64.64598529287485,
        ],
        result[16..]
    );
}

#[test]
fn test_obv() {
    let stats = common::test_data();
    let result = volume::obv(&stats.close, &stats.volume).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            24398800.0, 59729800.0, 40971500.0, 28363400.0, 15375500.0, 24818400.0, 19995500.0,
            15377100.0, 18304100.0, 15642600.0, 13365300.0, 9015200.0, 13278200.0, 11264800.0,
            7816300.0, 9515300.0, 7361500.0, 9645600.0, 7117500.0, 8603900.0, 10560400.0,
            11944400.0, 10458600.0, 13222800.0, 15901100.0, 14148200.0, 15746300.0, 18111300.0,
            16923000.0, 19039700.0, 25281400.0, 38772900.0, 36090498.0,
        ],
        result[1..]
    );
}

#[test]
fn test_bw_mfi() {
    let stats = common::test_data();
    let result = volume::bw_mfi(&stats.high, &stats.low, &stats.volume).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            0.3551550727197042,
            0.6500318374532476,
            0.33087097567026835,
            0.4339412094725235,
            0.49174742926685644,
            0.37611918181793846,
            0.7783624177023047,
            0.9724444714603765,
            0.7210293242366811,
            1.233344930082529,
            1.5555135786768504,
            1.1373117958055112,
            0.6615936075149493,
            1.018062433166289,
            0.8443432814837852,
            0.701753854993566,
            1.6597997026628717,
            0.8844826721593811,
            2.0927291248168385,
            1.4477289909850777,
            1.3858997398351827,
            1.1438807958597543,
            1.0982662267078556,
            1.2047388043662295,
            1.2839157030522224,
            1.131315608406131,
            1.4861084729512968,
            1.9398049488882794,
            1.6570819104700742,
            2.07018352644337,
            1.4739920309371974,
            0.46942344316064233,
            0.4046991872269693,
            1.0624799989416553,
        ],
        result
    );
}

#[test]
fn test_pvi() {
    let stats = common::test_data();
    let result = volume::pvi(&stats.close, &stats.volume).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            129.99999834143597,
            141.54347958772078,
            141.54347958772078,
            141.54347958772078,
            131.92931812027018,
            131.92931812027018,
            131.92931812027018,
            131.92931812027018,
            131.92931812027018,
            131.92931812027018,
            131.92931812027018,
            123.39788932917753,
            123.39788932917753,
            123.39788932917753,
            116.77112656052807,
            116.77112656052807,
            111.17950611133085,
            118.41739462626037,
            116.03315141809483,
            116.03315141809483,
            118.18599905113699,
            118.18599905113699,
            116.26872705625792,
            124.43081768468825,
            124.43081768468825,
            124.43081768468825,
            124.43081768468825,
            129.36682563893052,
            129.36682563893052,
            133.88801060311675,
            137.02232992266445,
            142.56979250121856,
            142.56979250121856,
        ],
        result[1..]
    );
}

#[test]
fn test_nvi() {
    let stats = common::test_data();
    let result = volume::nvi(&stats.close, &stats.volume).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            100.0,
            100.0,
            88.69605200221511,
            75.74873173474221,
            75.74873173474221,
            83.13081297218615,
            77.42947437108693,
            75.43564834164135,
            78.36871197877878,
            76.04533374620335,
            74.15037714917995,
            74.15037714917995,
            78.44894730261407,
            74.46748549955687,
            74.46748549955687,
            76.58980770001058,
            76.58980770001058,
            76.58980770001058,
            76.58980770001058,
            78.76309935666443,
            78.76309935666443,
            79.37010466242073,
            79.37010466242073,
            79.37010466242073,
            80.85512485614068,
            77.64048687003587,
            80.15629456805021,
            80.15629456805021,
            78.37504129842624,
            78.37504129842624,
            78.37504129842624,
            78.37504129842624,
            75.16531911680467,
        ],
        result[1..]
    );
}

#[test]
fn test_vwap() {
    let stats = common::test_data();
    let result = volume::vwap(
        &stats.high,
        &stats.low,
        &stats.close,
        &stats.volume,
        Some(&[7, 15, 23, 31]),
    )
    .collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            47.4466667175293,
            52.58425364741419,
            59.66093472503018,
            59.53824270598854,
            58.56073873512394,
            57.26083140126081,
            56.629122534293316,
            48.26000086466471,
            47.368080216184346,
            47.24389729733282,
            47.16352063096499,
            46.98366761685646,
            46.14386510953216,
            45.84006293015352,
            45.6182982182726,
            40.59333292643229,
            40.683550150424935,
            40.331958003437705,
            40.34817175614751,
            40.614995042842864,
            40.734309168125854,
            41.00676797780979,
            41.1851341756428,
            41.97999954223633,
            43.78789572143555,
            45.030899785559015,
            45.047881306449035,
            45.20523317347606,
            45.66930025458913,
            45.80022352517835,
            46.13153717323467,
            49.0433349609375,
            51.12404763373409,
            50.93548690748761
        ],
        result
    );
    assert_ne!(
        (stats.high[6] + stats.low[6] + stats.close[6]) / 3.0,
        result[6]
    );
    assert_eq!(
        (stats.high[7] + stats.low[7] + stats.close[7]) / 3.0,
        result[7]
    );
    assert_ne!(
        (stats.high[8] + stats.low[8] + stats.close[8]) / 3.0,
        result[8]
    );
}

#[test]
fn test_vwma() {
    let stats = common::test_data();
    let result = volume::vwma(&stats.close, &stats.volume, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            53.900419215672876,
            54.61133940105144,
            53.325953025570854,
            48.47630597789053,
            45.90374980207429,
            45.14058196286173,
            44.851298103743225,
            43.62068385486218,
            43.187592221412366,
            43.04208029753131,
            42.9266510253541,
            42.77282522869185,
            42.770014711447914,
            43.17763043835742,
            43.12727989018482,
            43.51073255270929,
            44.88017874003854,
            46.86682348303129,
            47.34784094060334,
        ],
        result[15..]
    );
}

#[test]
fn test_vpt() {
    let stats = common::test_data();
    let result = volume::vpt(&stats.close, &stats.volume).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            7319639.59533028,
            10456891.450067371,
            8336462.9727988895,
            6496007.789289162,
            5613821.141858923,
            6534077.538509425,
            6203309.860964381,
            6084385.043132608,
            6198191.67772293,
            6119286.831517547,
            6062539.304143893,
            5781232.852227728,
            6028363.147023435,
            5926178.5356308445,
            5740985.810439846,
            5789407.284515163,
            5686271.924960004,
            5834968.9441786,
            5784067.594139721,
            5826245.278291941,
            5862545.6488547465,
            5873211.752017631,
            5849108.366228603,
            6043155.824099114,
            6093267.00389101,
            6023575.207163605,
            6075358.914920703,
            6169175.374773634,
            6142768.674317974,
            6216744.3026634,
            6362862.545877969,
            6909077.10390515,
            6799223.698321057,
        ],
        result[1..]
    );
}
