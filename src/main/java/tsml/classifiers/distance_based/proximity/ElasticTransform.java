package tsml.classifiers.distance_based.proximity;

import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.ed.EDistanceConfigs;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.intervals.IntervalInstance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearchIterator;
import tsml.transformers.BaseTrainableTransformer;
import utilities.ArrayUtilities;
import weka.core.*;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Stream;

import static tsml.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_FLAG;
import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class ElasticTransform extends BaseTrainableTransformer {

    public void setRandom(final Random random) {
        this.random = random;
    }

    public boolean isRandomIntervals() {
        return randomIntervals;
    }

    public void setRandomIntervals(final boolean randomIntervals) {
        this.randomIntervals = randomIntervals;
    }

    public int getMinIntervalLength() {
        return minIntervalLength;
    }

    public void setMinIntervalLength(final int minIntervalLength) {
        this.minIntervalLength = minIntervalLength;
    }

    public static class Feature {
        private Feature(final Instance instance, final DistanceFunction distanceFunction,
                final Interval interval) {
            this.instance = instance;
            this.distanceFunction = distanceFunction;
            this.interval = interval;
        }

        private final Instance instance;
        private final DistanceFunction distanceFunction;
        private final Interval interval;

        public Instance getInstance() {
            return instance;
        }

        public DistanceFunction getDistanceFunction() {
            return distanceFunction;
        }

        public Interval getInterval() {
            return interval;
        }

        public double distance(Instance b) {
            Instance a = instance;
            if(a == b) {
                return 0;
            }
            if(interval != null) {
                a = new IntervalInstance(a, interval);
                b = new IntervalInstance(b, interval);
            }
            return distanceFunction.distance(a, b);
        }
    }

    private List<ParamSpaceBuilder> paramSpaceBuilders = newArrayList(
            i -> EDistanceConfigs.buildEdSpace(),
            DTWDistanceConfigs::buildDtwSpaceContinuousUnrestricted,
            i -> DTWDistanceConfigs.buildDtwFullWindowSpace(),
            DTWDistanceConfigs::buildDdtwSpaceContinuousUnrestricted,
            i -> DTWDistanceConfigs.buildDdtwFullWindowSpace(),
            LCSSDistanceConfigs::buildLcssSpaceContinuousUnrestricted,
            ERPDistanceConfigs::buildErpSpaceContinuousUnrestricted,
            i -> MSMDistanceConfigs.buildMsmSpaceContinuous(),
            i -> TWEDistanceConfigs.buildTwedSpaceContinuous(),
            i -> WDTWDistanceConfigs.buildWdtwSpaceContinuous(),
            i -> WDTWDistanceConfigs.buildWddtwSpaceContinuous()
    );
    private int numFeatures = 1000;
    private List<Feature> features;
    private Random random;
    private List<ParamSpace> paramSpaces;
    private boolean reset = true;
    private List<List<Double>> transformedTrain;
    private boolean randomIntervals = false;
    private int minIntervalLength = 3;

    public int getNumFeatures() {
        return numFeatures;
    }

    public void setNumFeatures(final int numFeatures) {
        this.numFeatures = numFeatures;
    }

    public Random getRandom() {
        return random;
    }

    @Override public void reset() {
        super.reset();
        reset = true;
    }

    @Override public void fit(final Instances trainData) {
        super.fit(trainData);
        if(reset) {
            reset = false;
            // build param spaces
            paramSpaces = new ArrayList<>(paramSpaceBuilders.size());
            for(ParamSpaceBuilder builder : paramSpaceBuilders) {
                final ParamSpace paramSpace = builder.build(trainData);
                paramSpaces.add(paramSpace);
            }
            // build transformed data container
            features = new ArrayList<>();
        }
        final int seriesLength = trainData.numAttributes() - 1;
        // add remaining features
        while(features.size() < numFeatures) {
            final int instanceIndex = random.nextInt(trainData.size());
            final Instance instance = trainData.get(instanceIndex);
            final int paramSpaceIndex = random.nextInt(paramSpaces.size());
            final ParamSpace paramSpace = paramSpaces.get(paramSpaceIndex);
            final ParamSet paramSet = RandomSearchIterator.choice(random, paramSpace);
            final DistanceFunction distanceFunction = (DistanceFunction) paramSet.getSingle(DISTANCE_MEASURE_FLAG);
            distanceFunction.setInstances(trainData);
            final int length = random.nextInt(seriesLength + 1 - minIntervalLength) + minIntervalLength;
            final int start = random.nextInt(seriesLength - length + 1);
            final Interval interval = new Interval(start, length);
            final Feature feature = new Feature(instance, distanceFunction, interval);
            features.add(feature);
        }
    }

    @Override public Instances fitTransform(final Instances trainData) {
        final boolean reset = this.reset;
        fit(trainData);
        if(reset) {
            transformedTrain = new ArrayList<>();
            for(int i = 0; i < trainData.size(); i++) {
                transformedTrain.add(newArrayList(trainData.get(i).classValue()));
            }
        }
        // for each remaining feature
        for(int i = transformedTrain.get(0).size() - 1; i < features.size(); i++) {
            final Feature feature = features.get(i);
            // make an attribute for the feature
            for(int j = 0; j < transformedTrain.size(); j++) {
                // find attribute value for each instance
                final List<Double> transformedInstance = transformedTrain.get(j);
                Instance b = trainData.get(j);
                final double distance = feature.distance(b);
                transformedInstance.add(i, distance);
            }
        }
        Instances result = determineOutputFormat(trainData);
        for(int i = 0; i < transformedTrain.size(); i++) {
            final List<Double> atts = transformedTrain.get(i);
            final double[] unboxedAtts = ArrayUtilities.unbox(atts);
            final Instance instance = new DenseInstance(trainData.get(i).weight(), unboxedAtts);
            result.add(instance);
        }
        return result;
    }

    @Override public Instance transform(final Instance inst) {
        final Instances output = determineOutputFormat(inst.dataset());
        final Instance transformed = transform(inst, output);
        output.add(transformed);
        return transformed;
    }

    public Instance transform(final Instance inst, Instances output) {
        if(!isFit()) throw new IllegalStateException("must fit first");
        double[] atts = new double[features.size() + 1];
        for(int i = 0; i < features.size(); i++) {
            final Feature feature = features.get(i);
            final double distance = feature.distance(inst);
            atts[i] = distance;
        }
        atts[atts.length - 1] = inst.classValue();
        final DenseInstance transformed = new DenseInstance(inst.weight(), atts);
        transformed.setDataset(output);
        return transformed;
    }

    @Override public Instances transform(final Instances data) {
        final Instances output = determineOutputFormat(data);
        for(Instance instance : data) {
            output.add(transform(instance, output));
        }
        return output;
    }

    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        ArrayList<Attribute> atts = new ArrayList<>();
        for(int i = 0; i < features.size(); i++) {
            atts.add(new Attribute("att" + i));
        }
        atts.add(data.classAttribute());
        final Instances output = new Instances("et", atts, 0);
        output.setClassIndex(output.numAttributes() - 1);
        return output;
    }

    public static void main(String[] args) {
        final ExecutorService executorService =
                Executors.newFixedThreadPool(1);//Runtime.getRuntime().availableProcessors());
        final String name = "eti";
        boolean randomIntervals = true;
        try (Stream<String> lines = Files.lines(Paths.get("/bench/phd/datasets/lists/2015_pigless.txt"), Charset.defaultCharset())) {
            lines.forEachOrdered(datasetName -> {
                List<Integer> sizes = newArrayList(10,20,50,100,200,500,1000);//,5000,10000);
                for(int i = 0; i < 30; i++) {
                    final int finalI = i;
//                    executorService.submit(() -> {
                        try {
                            final Instances[] instances = DatasetLoading.sampleDataset(
                                    "/bench/phd/datasets/uni2018", datasetName,
                                    finalI);
                            ElasticTransform et = new ElasticTransform();
                            et.setRandom(new Random(finalI));
                            et.setRandomIntervals(randomIntervals);
                            for(int numFeatures : sizes) {
                                System.out.println(datasetName + " " + finalI + " " + numFeatures + " start");
                                et.setNumFeatures(numFeatures);
                                final String s = "/bench/phd/datasets/" + name + numFeatures + "/" + datasetName + "/" + datasetName +
                                                         finalI;
                                final String trainPath = s + "_TRAIN.arff";
                                final String testPath = s + "_TEST.arff";
                                if(!new File(trainPath).exists()) {
                                    final Instances transformedTrain = et.fitTransform(instances[0]);
                                    writeInstancesToFile(transformedTrain, trainPath);
                                }
                                if(!new File(testPath).exists()) {
                                    final Instances transformedTest = et.transform(instances[1]);
                                    writeInstancesToFile(transformedTest, testPath);
                                }
                                System.out.println(datasetName + " " + finalI + " " + numFeatures + " done");
                            }
                        } catch(Exception e) {
                            e.printStackTrace();
                            System.exit(1);
                        }
//                    });
                }
            });
        } catch(IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        executorService.shutdown();
    }

    private static void writeInstancesToFile(Instances dataSet, String path) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataSet);
        saver.setFile(new File(path));
        saver.writeBatch();
    }

}
