package tsml.classifiers;

import experiments.data.DatasetLoading;
import fileIO.OutFile;
import org.apache.commons.lang3.ArrayUtils;
import tsml.classifiers.interval_based.TSF;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.GenericTools;
import utilities.generic_storage.Pair;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.StatisticalUtilities.dot;
import static utilities.Utilities.argMax;

/**
 * Implementation of the LEFTIST interpretability algorithm, producing model agnostic weights for time series interval
 * importance for new predictions.
 *
 * @author Matthew Middlehurst
 **/
public class LEFTIST {

    private TimeSeriesInstances sampleSeries;
    private EnhancedAbstractClassifier classifier;

    private int noSlices = 20;
    private int noNeighbours = 1000;

    public enum FeatureSelectionMethod{HIGHEST_WEIGHTS,FORWARD_SELECTION,NONE}
    private int noFeatures = 10;
    private FeatureSelectionMethod fsMethod = FeatureSelectionMethod.HIGHEST_WEIGHTS;

    private boolean distanceToSeries = false;
    private boolean predicictionClassOnly = false;
    private boolean generateScore = false;

    private Function<Pair<double[],double[]>,Double> similarityMeasure = (Pair<double[],double[]> p) ->
            -(dot(p.var1,p.var2) / (Math.sqrt(dot(p.var1,p.var1)) * Math.sqrt(dot(p.var2,p.var2)))) + 1;
    private Function<Double,Double> kernel = (Double d) ->
            Math.sqrt(Math.exp(-Math.pow(d, 2) / Math.pow(0.25, 2)));

    private LinearRegression explainer = new LinearRegression();

    private int seed;
    private Random rand;

    private boolean debug = false;

    public LEFTIST(TimeSeriesInstances sampleSeries, EnhancedAbstractClassifier classifier, int seed){
        this.sampleSeries = sampleSeries;
        this.classifier = classifier;
        this.seed = seed;
        this.rand = new Random(seed);
    }

    public LEFTIST(Instances sampleSeries, EnhancedAbstractClassifier classifier, int seed){
        this.sampleSeries = Converter.fromArff(sampleSeries);
        this.classifier = classifier;
        this.seed = seed;
        this.rand = new Random(seed);
    }

    public void setNoSlices(int i){
        noSlices = i;
    }

    public void setNoNeighbours(int i){
        noNeighbours = i;
    }

    public void setDistanceToSeries(boolean b) { distanceToSeries = b; }

    public void setDistanceMeasure(Function<Pair<double[],double[]>,Double> f) { similarityMeasure = f; }

    public void setKernel(Function<Double,Double> f) { kernel = f; }

    public void setDebug(boolean b) { debug = b; }

    public void setSeed(int i) {
        seed = i;
        rand = new Random(seed);
    }

    public Explanation generateExplanation(TimeSeriesInstance inst) throws Exception {
        if (inst.getNumDimensions() > 1){
            System.err.println("Only available for univariate series.");
            return null;
        }

        Explanation e = new Explanation();

        e.slices = slices(inst.getMaxLength());
        double[][] activatedSlices = neighbourSliceActivation();
        double[][] transformedNeighbours = transformNeighbours(inst, e.slices, activatedSlices);
        double[] neighbourWeights = weightNeighbours(distanceToSeries ? transformedNeighbours : activatedSlices);

        double[][] probas = new double[noNeighbours][];
        for (int i = 0; i < noNeighbours; i++) {
            double[][] dims = new double[1][];
            dims[0] = transformedNeighbours[i];
            probas[i] = classifier.distributionForInstance(new TimeSeriesInstance(dims));
        }
        e.predVal = argMax(probas[0], rand);

        int noClasses;
        if (predicictionClassOnly){
            noClasses = 1;
            e.classes = new int[]{ e.predVal };
        }
        else{
            noClasses = sampleSeries.getClassLabels().length;
            e.classes = new int[sampleSeries.numClasses()];
            for (int i = 0; i < e.classes.length; i++){
                e.classes[i] = i;
            }
        }

        e.coefficientsForClass = new double[noClasses][];
        e.classMeans = new double[noClasses];
        e.scores = new double[noClasses];
        Arrays.fill(e.scores, -1);

        for (int i = 0 ; i < noClasses; i++) {
            e.usedFeatures = featureSelection(activatedSlices, neighbourWeights, probas, e.classes[i]);

            Instances maskInstances = maskInstances(activatedSlices, neighbourWeights, e.usedFeatures);
            for (int n = 0; n < noNeighbours; n++){
                maskInstances.get(n).setClassValue(probas[n][e.classes[i]]);
            }

            LinearRegression lr = (LinearRegression) AbstractClassifier.makeCopy(explainer);
            lr.buildClassifier(maskInstances);

            if (generateScore){
                e.scores[i] = score(lr, maskInstances);
            }

            double[] c = lr.coefficients();
            e.coefficientsForClass[i] = new double[noSlices];
            for (int n = 0; n < noFeatures; n++){
                e.coefficientsForClass[i][e.usedFeatures[n]] = c[n];
            }
            e.classMeans[i] = c[c.length-1];
        }

        return e;
    }

    public Explanation generateExplanation(Instance inst) throws Exception {
        return generateExplanation(Converter.fromArff(inst));
    }

    public void outputFigure(TimeSeriesInstance inst, String figureSavePath) throws Exception {
        Explanation exp = generateExplanation(inst);

        File f = new File(figureSavePath);
        if(!f.isDirectory()) f.mkdirs();

        OutFile of = new OutFile(figureSavePath + "\\interp" + seed + ".txt");
        of.writeLine(Arrays.toString(inst.toValueArray()[0]));
        of.writeString(Arrays.toString(exp.slices[0]));
        for (int i = 1; i < exp.slices.length; i++) {
            of.writeString(";" + Arrays.toString(exp.slices[i]));
        }
        of.writeLine("");;
        of.writeLine(Integer.toString(exp.predVal));
        of.writeLine(Integer.toString(sampleSeries.getClassLabels().length));
        for (int i = 0; i < exp.coefficientsForClass.length; i++){
            of.writeLine(Integer.toString(exp.classes[i]));
            of.writeLine(Arrays.toString(exp.coefficientsForClass[i]));
        }

        Process p = Runtime.getRuntime().exec("py src/main/python/leftist.py \"" +
                figureSavePath.replace("\\", "/")+ "\" " + seed);

        if (debug) {
            System.out.println("LEFTIST interp python output:");
            BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader err = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            System.out.println("output : ");
            String outLine = out.readLine();
            while (outLine != null) {
                System.out.println(outLine);
                outLine = out.readLine();
            }
            System.out.println("error : ");
            String errLine = err.readLine();
            while (errLine != null) {
                System.out.println(errLine);
                errLine = err.readLine();
            }
        }
    }

    public void outputFigure(Instance inst, String figureSavePath) throws Exception {
        outputFigure(Converter.fromArff(inst), figureSavePath);
    }

    private int[][] slices(int length){
        if (length < noSlices) noSlices = length;
        int[][] slices = new int[noSlices][2];
        int sliceSize = length / noSlices;
        int remainder = length % noSlices;
        int sum = 0;
        for (int i = 0; i < noSlices; i++){
            slices[i][0] = sum;
            sum += sliceSize;
            if (remainder > 0){
                sum++;
                remainder--;
            }
            slices[i][1] = sum;
        }
        return slices;
    }

    private double[][] neighbourSliceActivation(){
        double[][] activatedSlices = new double[noNeighbours][noSlices];
        Arrays.fill(activatedSlices[0], 1);
        for (int i = 1; i < noNeighbours; i++){
            for (int n = 0; n < noSlices; n++){
                activatedSlices[i][n] = rand.nextBoolean() ? 1 : 0;
            }
        }
        return activatedSlices;
    }

    private double[][] transformNeighbours(TimeSeriesInstance inst, int[][] slices, double[][] activatedSlices){
        double[] instVals = inst.toValueArray()[0];
        double[][] transformedNeighbours = new double[noNeighbours][];
        for (int i = 0; i < noNeighbours; i++){
            transformedNeighbours[i] = sampleSeries.get(rand.nextInt(sampleSeries.numInstances())).toValueArray()[0];
            for (int n = 0 ; n < noSlices; n++){
                if (activatedSlices[i][n] == 1){
                    System.arraycopy(instVals, slices[n][0], transformedNeighbours[i], slices[n][0],
                            slices[n][1] - slices[n][0]);
                }
            }
        }
        return transformedNeighbours;
    }

    private double[] weightNeighbours(double[][] neighbours){
        double[] weights = new double[neighbours.length];
        Pair<double[],double[]> p = new Pair<>(neighbours[0], null);
        for (int i = 0; i < noNeighbours; i++){
            p.var2 = neighbours[i];
            weights[i] = kernel.apply(similarityMeasure.apply(p));
        }
        return weights;
    }

    private Instances maskInstances(double[][] activatedSlices, double[] neighbourWeights, int[] usedFeatures){
        int noAtts = usedFeatures.length;

        ArrayList<Attribute> atts = new ArrayList<>(noAtts + 1);
        ArrayList<String> vals = new ArrayList<>(2);
        vals.add("0");
        vals.add("1");
        for (int i = 0; i < noAtts; i++) {
            atts.add(new Attribute(Integer.toString(i), vals));
        }
        atts.add(new Attribute("class"));

        Instances maskInstances = new Instances("maskInstances", atts, noNeighbours);
        maskInstances.setClassIndex(maskInstances.numAttributes()-1);
        for (int i = 0; i < noNeighbours; i++) {
            double[] newArr = new double[noAtts + 1];
            for (int n = 0; n < noAtts; n++){
                newArr[n] = activatedSlices[i][usedFeatures[n]];
            }
            maskInstances.add(new DenseInstance(neighbourWeights[i], newArr));
        }

        return maskInstances;
    }

    private int[] featureSelection(double[][] activatedSlices, double[] neighbourWeights, double[][] probas, int cls)
            throws Exception {
        int[] usedFeatures;

        if (fsMethod == FeatureSelectionMethod.NONE || noFeatures >= noSlices){
            usedFeatures = new int[noSlices];
            for (int i = 0; i < noSlices; i++) {
                usedFeatures[i] = i;
            }
        }
        else if (fsMethod == FeatureSelectionMethod.HIGHEST_WEIGHTS) {
            usedFeatures = new int[noFeatures];

            Integer[] allFeatures = new Integer[noSlices];
            for (int i = 0; i < noSlices; i++) {
                allFeatures[i] = i;
            }

            Instances maskInstances = maskInstances(activatedSlices, neighbourWeights, Arrays.stream(allFeatures)
                    .mapToInt(i -> i).toArray());
            for (int n = 0; n < noNeighbours; n++){
                maskInstances.get(n).setClassValue(probas[n][cls]);
            }

            LinearRegression lr = new LinearRegression();
            lr.buildClassifier(maskInstances);

            double[] c = new double[noSlices];
            System.arraycopy(lr.coefficients(), 0, c, 0, noSlices);
            Arrays.sort(allFeatures, new GenericTools.SortIndexDescending(c));

            for (int i = 0; i < noFeatures; i++) {
                usedFeatures[i] = allFeatures[i];
            }
        }
        else if (fsMethod == FeatureSelectionMethod.FORWARD_SELECTION) {
            usedFeatures = new int[noFeatures];
            Arrays.fill(usedFeatures, -1);

            ArrayList<String> vals = new ArrayList<>(2);
            vals.add("0");
            vals.add("1");

            ArrayList<Attribute> atts = new ArrayList<>(1);
            atts.add(new Attribute("class"));
            Instances fsInstances = new Instances("fs", atts, noNeighbours);
            for (int i = 0; i < noNeighbours; i++){
                double[] a = new double[]{probas[i][cls]};
                fsInstances.add(new DenseInstance(neighbourWeights[i], a));
            }

            for (int i = 0; i < noFeatures; i++) {
                double max = -99999999;
                int feature = 0;

                fsInstances.insertAttributeAt(new Attribute(Integer.toString(i), vals), i);

                for (int n = 0; n < noSlices; n++){
                    if (ArrayUtils.contains(usedFeatures, n)) continue;

                    for (int g = 0; g < noNeighbours; g++){
                        fsInstances.get(g).setValue(i, activatedSlices[g][n]);
                    }

                    LinearRegression lr = new LinearRegression();
                    lr.buildClassifier(fsInstances);
                    double score = score(lr, fsInstances);

                    if (score > max){
                        max = score;
                        feature = n;
                    }
                }

                usedFeatures[i] = feature;
            }
        }
        else {
            throw new Exception("Invalid feature selection option.");
        }

        return usedFeatures;
    }

    private double score(Classifier cls, Instances insts) throws Exception {
        double[] preds = new double[insts.numInstances()];
        int mean = 0;
        int weightSum = 0;
        for (int i = 0 ; i < preds.length; i++){
            Instance inst = insts.get(i);

            preds[i] = cls.classifyInstance(inst);
            mean += inst.classValue() * inst.weight();
            weightSum += inst.weight();
        }
        mean /= weightSum;

        double n = 0;
        double d = 0;
        for (int i = 0; i < preds.length; i++){
            Instance inst = insts.get(i);

            n += inst.weight() * Math.pow(inst.classValue() - preds[i], 2);
            d += inst.weight() * Math.pow(inst.classValue() - mean, 2);
        }

        return d != 0 ? 1 - n / d : 0;
    }

    public static class Explanation {
        int predVal;
        int[][] slices;
        int[] classes;
        int[] usedFeatures;
        double[][] coefficientsForClass;
        double[] classMeans;
        double[] scores;

        public Explanation() {}

        @Override
        public String toString(){
            return predVal + "\n" + Arrays.deepToString(slices) + "\n" + Arrays.toString(classes) + "\n" +
                    Arrays.toString(usedFeatures) + "\n" + Arrays.deepToString(coefficientsForClass) + "\n" +
                    Arrays.toString(classMeans) + "\n" + Arrays.toString(scores);
        }
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\" + dataset +
                "\\" + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\" + dataset +
                "\\" + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        TSF c = new TSF();
        c.seed = 0;
        c.buildClassifier(train);
        LEFTIST l = new LEFTIST(train, c, 0);
        l.outputFigure(test.get(0), "E:\\Temp\\LEFTIST\\" + dataset + "1\\");
        l.outputFigure(test.get(1), "E:\\Temp\\LEFTIST\\" + dataset + "2\\");
        l.outputFigure(test.get(2), "E:\\Temp\\LEFTIST\\" + dataset + "3\\");
        l.outputFigure(test.get(3), "E:\\Temp\\LEFTIST\\" + dataset + "4\\");
        l.outputFigure(test.get(4), "E:\\Temp\\LEFTIST\\" + dataset + "5\\");
    }
}
