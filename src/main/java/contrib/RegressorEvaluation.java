package contrib;

import evaluation.storage.EstimatorResultsCollection;

import static contrib.ClassifierEvaluation.summariseResultsPresent;

public class RegressorEvaluation {
    static String[] allRegressionProblems = {
            "AppliancesEnergy",
            "AustraliaRainfall",
            "BeijingPM10Quality-no-missing",
            "BeijingPM25Quality-no-missing",
            "BenzeneConcentration-no-missing",
            "BIDMC32HR",
            "BIDMC32RR",
            "BIDMC32SpO2",
            "Covid3Month",
            "FloodModeling1",
            "FloodModeling2",
            "FloodModeling3",
            "HouseholdPowerConsumption1-no-missing",
            "HouseholdPowerConsumption2-no-missing",
            "IEEEPPG",
            "LiveFuelMoistureContent",
            "NewsHeadlineSentiment",
            "NewsTitleSentiment",
            "PPGDalia-equal-length",
            "Covid19Andalusia",
            "BeijingIntAirportPM25Quality",
            "ParkingBirmingham-equal-length",
            "TetuanEnergyConsumption",
            "BarCrawl6min",
            "MetroInterstateTrafficVolume",
            "ASHRAEGreatEnergyPredictorHotwater",
            "ASHRAEGreatEnergyPredictorSteam",
            "ASHRAEGreatEnergyPredictorChilledWater",
            "ASHRAEGreatEnergyPredictorElectricity",
            "MadridPM10Quality-no-missing",
// Dont use duplicate of  BenzeneConcentration           "AirQuality",
            "GasSensorArrayEthanol",
            "GasSensorArrayAcetone",
            "SierraNevadaMountainsSnow",
            "BitcoinSentiment",
            "EthereumSentiment",
            "CardanoSentiment",
            "BinanceCoinSentiment",
            "ElectricMotorTemperature",
            "USASouthwestEnergyFlux",
            "USASouthwestSWH",
            "SolarRadiationAndalusia-no-missing",
            "PrecipitationAndalusia-no-missing",
            "AcousticContaminationMadrid-no-missing",
            "VentilatorPressure",
            "OccupancyDetectionLight",
            "WindTurbinePower",
            "DhakaHourlyAirQuality",
            "DailyOilGasPrices",
            "WaveDataTension",
            "NaturalGasPricesSentiment",
            "DailyTemperatureLatitude",
            "MethaneMonitoringHomeActivity",
            "LPGasMonitoringHomeActivity",
            "AfricaSoilAlphaKBrAluminium",
            "AfricaSoilAlphaKBrBoron",
            "AfricaSoilAlphaKBrCopper",
            "AfricaSoilAlphaZnSeIron",
            "AfricaSoilAlphaZnSeManganese",
            "AfricaSoilAlphaZnSeSodium",
            "AfricaSoilHTSXTPhosphorus",
            "AfricaSoilHTSXTPotassium",
            "AfricaSoilHTSXTMagnesium",
            "AfricaSoilMPASulphur",
            "AfricaSoilMPAZinc",
            "AfricaSoilMPACalcium"
    };

    static String[] allRegressors={
            //Already in sktime
//            "CNNRegressor",
//            "KNeighborsTimeSeriesRegressor",
 //           "RocketRegressor",
 //           "TimeSeriesForestRegressor",
            //sktime in tsml-eval
            "1nn-ed",
            "5nn-ed",
            "1nn-dtw",
            "5nn-dtw",
            "cnn",
            "DrCIF",
            "DrCIF-d",
            "DrCIF-p",
            "DrCIF-s",
            "fcn",
            "fpcr",
            "fpcr-b-spline",
            "fresh-prince",
            "grid-svr",
            "hydra",
            "inception",
            "lr",
            "minirocket",
            "multirocket",
            "ResNetRegressor",
            "rf",
            "ridge",
            "rocket",
            "rotF",
            "svr",
            "SingleInception",
            "TimeSeriesForestRegressor",
            "xgb"
    };

    static String[] monashTSERRegressors = {
            "1nn-dtw",
            "5nn-dtw",
            "rocket",
            "fpcr",
            "fpcr-b-spline",
            "grid-svr",
            "rf",
            "xgb",
            "inception",
            "fcn",
//            "resnet"
    };

    static String[] monashTSERRegressorsReduced = {
            "1nn-ed",
            "5nn-ed",
            "rocket",
            "fpcr",
            "fpcr-b-spline",
            "grid-svr",
            "rf",
            "xgb",
            "inception",
            "fcn",
//            "resnet"
    };

    static String[] monashTSERProblems =    {
        "AppliancesEnergy",
                "AustraliaRainfall",
                "BeijingPM10Quality-no-missing",
                "BeijingPM25Quality-no-missing",
                "BenzeneConcentration-no-missing",
                "BIDMC32HR",
                "BIDMC32RR",
                "BIDMC32SpO2",
                "Covid3Month",
                "FloodModeling1",
                "FloodModeling2",
                "FloodModeling3",
                "HouseholdPowerConsumption1-no-missing",
                "HouseholdPowerConsumption2-no-missing",
                "IEEEPPG",
                "LiveFuelMoistureContent",
                "NewsHeadlineSentiment",
                "NewsTitleSentiment",
                "PPGDalia-equal-length"
    };

    public static void recreateMonash() throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Regression\\Analysis\\", "monashRecreationAllProblems", 1);
//        String[] datasets = monashTSERProblems;
        String[] datasets = allRegressionProblems;
        x.setDatasets(datasets);
        x.setUseRegressionStatistics();
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setResultsType(EstimatorResultsCollection.ResultsType.REGRESSION);
        x.setTestResultsOnly(true);
        //x.setDebugPrinting(true);
        String c1 = "C:\\Results Working Area\\Regression\\sktime\\";
        x.readInEstimators(monashTSERRegressors, c1);
        x.runComparison();
    }


    public static void regressorCompare(String[] regressors, String[] problems,String experimentName, int resamples) throws Exception {
            evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Regression\\Analysis\\", experimentName, resamples);
            String[] datasets = problems;
            x.setDatasets(datasets);
            x.setUseRegressionStatistics();
            x.setIgnoreMissingResults(true);
            x.setBuildMatlabDiagrams(true);
            x.setResultsType(EstimatorResultsCollection.ResultsType.REGRESSION);
            x.setTestResultsOnly(true);
            //x.setDebugPrinting(true);
            String c1 = "C:\\Results Working Area\\Regression\\sktime\\";
            x.readInEstimators(regressors, c1);
            x.runComparison();
        }
    static String[] goodRegressors = {
            "rocket",
            "rf",
            "xgb",
            "inception",
            "DrCIF",
            "fresh-prince",
//            "minirocket",
            "multirocket"
//            "resnet"
    };

    public static void main(String[] args) throws Exception {
//        summariseResultsPresent(allRegressors, allRegressionProblems, "X:\\Results Working Area\\Regression\\sktime\\", "regressionCounts.csv");
//        recreateMonash();
//        regressorCompare(goodRegressors,allRegressionProblems,"GoodRegressors",5);
//        regressorCompare(monashTSERRegressors,monashTSERProblems,"MonashRecreate16",1);
//        regressorCompare(monashTSERRegressors,allRegressionProblems,"MonashRecreate65",1);
        regressorCompare(goodRegressors,allRegressionProblems,"GoodRegressors65",5);
//        regressorCompare(monashTSERRegressorsReduced,monashTSERProblems,"MonashRecreate19",1);

    }
}
