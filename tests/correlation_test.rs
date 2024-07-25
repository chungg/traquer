use traquer::correlation;

mod common;

#[test]
fn test_pcc() {
    let stats = common::test_data();
    let stats2 = common::test_data_path("./tests/sp500.input");
    let result = correlation::pcc(&stats.close, &stats2.close, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            0.4949915809947665,
            0.567273447168426,
            0.6175013859936266,
            0.7735429829194097,
            0.8366952428366368,
            0.8096169725403108,
            0.809368028926954,
            0.8260221414870411,
            0.7993822738086296,
            0.7697660725871418,
            0.6969990163646897,
            0.570965450892101,
            0.3685491874676054,
            0.3921833351280792,
            0.48135394339924675,
            0.6753460541993099,
            0.7882896679836985,
            0.8428569396422592,
            0.869595477176004
        ],
        result[16 - 1..]
    );
}

#[test]
fn test_rsq() {
    let stats = common::test_data();
    let stats2 = common::test_data_path("./tests/sp500.input");
    let result = correlation::rsq(&stats.close, &stats2.close, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        correlation::pcc(&stats.close, &stats2.close, 16)
            .map(|x| x.powi(2))
            .collect::<Vec<_>>()[16 - 1..],
        result[16 - 1..]
    );
}

#[test]
fn test_beta() {
    let stats = common::test_data();
    let stats2 = common::test_data_path("./tests/sp500.input");
    let result = correlation::beta(&stats.close, &stats2.close, 8).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            4.355108264592213,
            4.601876992802109,
            4.327262701779409,
            4.775886060668173,
            4.794049255171498,
            3.7730000333307157,
            2.5643077783020236,
            2.633042328026424,
            2.514260517613367,
            3.1161159387423,
            3.0506307031608633,
            2.789878877605198,
            2.5955046321345887,
            2.7166741278782482,
            1.9669192450071182,
            2.0265952590234844,
            1.8919622781835297,
            1.1150261469480356,
            0.8646571690599486,
        ],
        result[8 + 8 - 1..]
    );
}

#[test]
fn test_perf() {
    let stats = common::test_data();
    let stats2 = common::test_data_path("./tests/sp500.input");
    let result = correlation::perf(&stats.close, &stats2.close, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            0.8478737539542476,
            0.8774178345363791,
            0.8613545956899847,
            0.9468798073031444,
            0.9550194431685893,
            0.9808040867527692,
            0.9895477133457892,
            1.0058379652941736,
            0.9986450679736525,
            1.0578756739724984,
            1.0750970259462438,
            1.0493240714551837,
            1.0831034127604227,
            1.1054918475884703,
            1.0633290357825254,
            1.0806454673215915,
            1.0914781496150827,
            1.12131946720817,
            1.0575579956667829
        ],
        result[16 - 1..]
    );
}

#[test]
fn test_rsc() {
    let stats = common::test_data();
    let stats2 = common::test_data_path("./tests/sp500.input");
    let result = correlation::rsc(&stats.close, &stats2.close).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            0.008788386806561178,
            0.011459912327557982,
            0.01251253937343318,
            0.011003164228403481,
            0.009386508089140053,
            0.008766593700616273,
            0.009691095179251999,
            0.009016615126422123,
            0.00889413859677848,
            0.009138527229425037,
            0.008870922847586863,
            0.008637385031681658,
            0.008155964965104049,
            0.008565009497378536,
            0.008250364341062738,
            0.007902296286528466,
            0.008144260338619862,
            0.007799354959989251,
            0.008325484172213089,
            0.008229939267731962,
            0.008390212768165614,
            0.008444843526680087,
            0.008508113273823885,
            0.008408571675862769,
            0.008907913131307135,
            0.009045829103802703,
            0.008825006929298515,
            0.009142374310920683,
            0.009419059066245307,
            0.00909553609534596,
            0.009317201456557284,
            0.009522524368411988,
            0.009908109478812252,
            0.009454208111562406
        ],
        result
    );
}

#[test]
fn test_krcc() {
    let stats = common::test_data();
    let stats2 = common::test_data_path("./tests/sp500.input");
    let result = correlation::krcc(&stats.close, &stats2.close, 16).collect::<Vec<_>>();
    assert_eq!(stats.close.len(), result.len());
    assert_eq!(
        vec![
            0.45,
            0.55,
            0.6166666666666667,
            0.6833333333333333,
            0.6666666666666666,
            0.6166666666666667,
            0.65,
            0.65,
            0.6,
            0.5833333333333334,
            0.5166666666666667,
            0.43333333333333335,
            0.2833333333333333,
            0.31666666666666665,
            0.36666666666666664,
            0.5,
            0.6166666666666667,
            0.6833333333333333,
            0.7166666666666667
        ],
        result[16 - 1..]
    );
    let result = correlation::krcc(
        &[7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4],
        &[2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0],
        7,
    )
    .collect::<Vec<_>>();
    assert_eq!(0.55, result[7 - 1]);
    let result = correlation::krcc(
        &[7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4],
        &[2.8, 2.8, 2.8, 2.6, 3.5, 4.6, 5.0],
        7,
    )
    .collect::<Vec<_>>();
    assert_eq!(0.6324555320336759, result[7 - 1]);
}
