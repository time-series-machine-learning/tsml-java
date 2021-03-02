/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package tsml.classifiers.multivariate;

import evaluation.evaluators.CrossValidationEvaluator;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.ensembles.ContractRotationForest;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.ShapeletTransform;
import tsml.transformers.shapelet_tools.ShapeletTransformFactory;
import tsml.transformers.shapelet_tools.ShapeletTransformFactoryOptions.ShapeletTransformOptions;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.concurrent.TimeUnit;

import static tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities.nanoToOp;
import static utilities.InstanceTools.resampleTrainAndTestInstances;

/**
 *
 *
 */
public class STC_D extends EnhancedAbstractClassifier {

    private ContractRotationForest classifier;
    private ShapeletTransform transform;

    private int[] redundantFeatures;
    private long transformBuildTime;
    private String[] classLabels;

    public STC_D(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        classifier = new ContractRotationForest();
        classifier.setMaxNumTrees(200);
    }

    @Override
    public String getParameters(){
        String paras=transform.getParameters();
        String ens=classifier.getParameters();
        return super.getParameters()+",TransformBuildTime,"+transformBuildTime+
                ",TransformParas,"+paras+",EnsembleParas,"+ens;
    }

    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        long startTime = System.nanoTime();

        classLabels = data.getClassLabels();

        Instances shapeletData = createTransformData(data);
        transformBuildTime = System.nanoTime()-startTime;

        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(shapeletData);

        if(getEstimateOwnPerformance()){
            int numFolds = setNumberOfFolds(data);
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            cv.setSeed(seed * 12);
            cv.setNumFolds(numFolds);
            trainResults = cv.crossValidateWithStats(classifier, shapeletData);
        }

        if (seedClassifier)
            classifier.setSeed(seed);

        classifier.buildClassifier(shapeletData);

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        if(getEstimateOwnPerformance()){
            trainResults.setBuildTime(System.nanoTime()-startTime - trainResults.getErrorEstimateTime());
        }
        else{
            trainResults.setBuildTime(System.nanoTime()-startTime);
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime()+trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        buildClassifier(Converter.fromArff(data));
    }

    @Override
    public double classifyInstance(TimeSeriesInstance ins) throws Exception{
        Instances temp = Converter.toArff(transform.transform(ins), classLabels).dataset();

        for(int del: redundantFeatures)
            temp.deleteAttributeAt(del);

        Instance test  = temp.get(0);
        return classifier.classifyInstance(test);
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception {
        return classifyInstance(Converter.fromArff(ins));
    }

    @Override
    public double[] distributionForInstance(TimeSeriesInstance ins) throws Exception{
        Instances temp = Converter.toArff(transform.transform(ins), classLabels).dataset();

        for(int del: redundantFeatures)
            temp.deleteAttributeAt(del);

        Instance test  = temp.get(0);
        return classifier.distributionForInstance(test);
    }

    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        return distributionForInstance(Converter.fromArff(ins));
    }

    public Instances createTransformData(TimeSeriesInstances train){
        int n = train.numInstances();
        int m = train.getMaxLength();
        int d = train.getMaxNumDimensions();

        ShapeletTransformOptions transformOptions=new ShapeletTransformOptions();
        transformOptions.setDistanceType(ShapeletDistance.DistanceType.DEPENDENT);
        transformOptions.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
        transformOptions.setRescalerType(ShapeletDistance.RescalerType.NORMALISATION);
        transformOptions.setRoundRobin(true);
        transformOptions.setCandidatePruning(true);
        transformOptions.setMinLength(3);
        transformOptions.setMaxLength(m);

        if(train.numClasses() > 2) {
            transformOptions.setBinaryClassValue(true);
            transformOptions.setClassBalancing(true);
        }else{
            transformOptions.setBinaryClassValue(false);
            transformOptions.setClassBalancing(false);
        }

        int numShapeletsInTransform = Math.min(10 * train.numInstances(), 2000);

        long transformContractTime = TimeUnit.NANOSECONDS.convert(4, TimeUnit.HOURS);

        SearchType searchType = SearchType.RANDOM;
        long numShapeletsInProblem = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n, m,
                3, m);

        double proportionToEvaluate = estimatePropOfFullSearch(n, m, d, transformContractTime);
        long numShapeletsToEvaluate;
        if(proportionToEvaluate == 1.0) {
            searchType = ShapeletSearch.SearchType.FULL;
            numShapeletsToEvaluate = numShapeletsInProblem;
        }
        else
            numShapeletsToEvaluate = (long) (numShapeletsInProblem * proportionToEvaluate);

        if(numShapeletsToEvaluate < n)
            numShapeletsToEvaluate = n;

        numShapeletsInTransform =  numShapeletsToEvaluate > numShapeletsInTransform ? numShapeletsInTransform :
                (int) numShapeletsToEvaluate;
        transformOptions.setKShapelets(numShapeletsInTransform);

        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        if(seedClassifier)
            searchBuilder.setSeed(2 * seed);
        searchBuilder.setMin(transformOptions.getMinLength());
        searchBuilder.setMax(transformOptions.getMaxLength());
        searchBuilder.setSearchType(searchType);
        searchBuilder.setNumShapeletsToEvaluate(numShapeletsToEvaluate / train.numInstances());

        transformOptions.setSearchOptions(searchBuilder.build());

        transform = new ShapeletTransformFactory(transformOptions.build()).getTransform();
        transform.setContractTime(transformContractTime);
        transform.setAdaptiveTiming(true);
        transform.setTimePerShapelet((double) transformContractTime / numShapeletsToEvaluate);
        transform.setPruneMatchingShapelets(false);

        return Converter.toArff(transform.fitTransform(train));
    }

    // Aarons way of doing it based on time for a single operation
    private double estimatePropOfFullSearch(int n, int m, int d, long time){
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        opCount = opCount.multiply(BigInteger.valueOf(d));

        double p = 1;
        if(opCount.compareTo(opCountTarget) > 0){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            p = prop.doubleValue();
        }

        return p;
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        String dataset = "ERing";
        Instances train = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\" +
                "MultivariateARFF\\" + dataset + "\\" + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\" +
                "MultivariateARFF\\" + dataset + "\\" + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        STC_D stc = new STC_D();
        stc.setSeed(fold);
        stc.setEstimateOwnPerformance(true);
        stc.buildClassifier(train);
        double acc = ClassifierTools.accuracy(test, stc);
        System.out.println("Test Accuracy = " + acc);
        System.out.println("Train Accuracy = "+ stc.trainResults.getAcc());
    }
}