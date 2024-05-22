use traquer::volume;

mod common;

#[test]
fn test_twiggs() {
    let stats = common::test_data();
    let result = volume::twiggs(&stats.high, &stats.low, &stats.close, &stats.volume, 16);
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
fn test_kvo() {
    let stats = common::test_data();
    let result = volume::kvo(&stats.high, &stats.low, &stats.close, &stats.volume, 10, 16);
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
fn test_elder_force() {
    let stats = common::test_data();
    let result = volume::elder_force(&stats.close, &stats.volume, 16);
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
    let result = volume::mfi(&stats.high, &stats.low, &stats.close, &stats.volume, 16);
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
    let result = volume::ad(&stats.high, &stats.low, &stats.close, &stats.volume);
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
    let result = volume::ad_yahoo(&stats.high, &stats.low, &stats.close, &stats.volume);
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
    let result = volume::cmf(&stats.high, &stats.low, &stats.close, &stats.volume, 16);
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
fn test_tvi() {
    let stats = common::test_data();
    let result = volume::tvi(&stats.close, &stats.volume, 0.5);
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
