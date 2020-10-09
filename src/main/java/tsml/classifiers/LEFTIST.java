package tsml.classifiers;

import experiments.data.DatasetLoading;
import fileIO.OutFile;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.dictionary_based.MultivariateIndividualTDE;
import tsml.classifiers.dictionary_based.TDE;
import tsml.classifiers.interval_based.CIF;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
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

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.Utilities.argMax;
import static weka.core.Utils.sum;

public class LEFTIST {

    private TimeSeriesInstances train;
    private EnhancedAbstractClassifier classifier;

    private int noSlices = 20;
    private int noNeighbours = 1000;
    private LinearRegression explainer = new LinearRegression();

    private int seed;
    private Random rand;

    private boolean debug = true;

    int predVal;
    int[][] predSlices;

    public LEFTIST(TimeSeriesInstances train, EnhancedAbstractClassifier classifier, int seed){
        this.train = train;
        this.classifier = classifier;
        this.seed = seed;
        this.rand = new Random(seed);
    }

    public LEFTIST(Instances train, EnhancedAbstractClassifier classifier, int seed){
        this.train = Converter.fromArff(train);
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

    public double[][] sliceWeights(TimeSeriesInstance inst) throws Exception {
        if (inst.getNumDimensions() > 1){
            System.err.println("Only available for univariate series.");
            return null;
        }

        int[][] slices = slices(inst.getMaxLength());
        predSlices = slices;
        double[][] activatedSlices = neighbourSliceActivation();
        double[][] transformedNeighbours = transformNeighbours(inst, slices, activatedSlices);
        double[] neighbourWeights = weightNeighbours(activatedSlices); //neighbours or masks?

        double[][] probas = new double[noNeighbours][];
        for (int i = 0; i < noNeighbours; i++) {
            double[][] dims = new double[1][];
            dims[0] = transformedNeighbours[i];
            probas[i] = classifier.distributionForInstance(new TimeSeriesInstance(dims));
        }
        predVal = argMax(probas[0], rand);

        //feature selection?

        int noClasses = train.getClassLabels().length;
        double[][] coeffs = new double[noClasses][];

        ArrayList<Attribute> atts = new ArrayList<>(noSlices + 1);
        ArrayList<String> vals = new ArrayList<>(2);
        vals.add("0");
        vals.add("1");
        for (int i = 0; i < noSlices; i++) {
            atts.add(new Attribute(Integer.toString(i), vals));
        }
        atts.add(new Attribute("class"));
        Instances maskInstances = new Instances("maskInstances", atts, noNeighbours);
        maskInstances.setClassIndex(maskInstances.numAttributes()-1);
        for (int i = 0; i < noNeighbours; i++) {
            double[] newArr = new double[noSlices + 1];
            System.arraycopy(activatedSlices[i], 0, newArr, 0, noSlices);
            maskInstances.add(new DenseInstance(neighbourWeights[i], newArr));
        }

        for (int i = 0 ; i < noClasses; i++) {
            for (int n = 0; n < noNeighbours; n++){
                maskInstances.get(n).setClassValue(probas[n][i]);
            }

            explainer.buildClassifier(maskInstances);

            //score?

            double[] c = explainer.coefficients();
            coeffs[i] = new double[noSlices];
            System.arraycopy(c, 0, coeffs[i], 0, noSlices);
        }

        return coeffs;
    }

    public double[][] sliceWeights(Instance inst) throws Exception {
        return sliceWeights(Converter.fromArff(inst));
    }

    public void outputFigure(TimeSeriesInstance inst, String figureSavePath) throws Exception {
        double[][] weights = sliceWeights(inst);

        File f = new File(figureSavePath);
        if(!f.isDirectory()) f.mkdirs();

        OutFile of = new OutFile(figureSavePath + "\\interp" + seed + ".txt");
        of.writeLine(Arrays.toString(inst.toValueArray()[0]));
        of.writeString(Arrays.toString(predSlices[0]));
        for (int i = 1; i < predSlices.length; i++) {
            of.writeString(";" + Arrays.toString(predSlices[i]));
        }
        of.writeLine("");;
        of.writeLine(Integer.toString(predVal));
        of.writeLine(Integer.toString(train.getClassLabels().length));
        for (int i = 0; i < train.getClassLabels().length; i++){
            of.writeLine(Integer.toString(i));
            of.writeLine(Arrays.toString(weights[i]));
        }

        Process p = Runtime.getRuntime().exec("python src/main/python/leftist.py \"" +
                figureSavePath.replace("\\", "/")+ "\" " + seed);

        if (debug) {
            System.out.println("CIF interp python output:");
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
            transformedNeighbours[i] = train.get(rand.nextInt(train.numInstances())).toValueArray()[0]; //should we be sampling from train?
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
        for (int i = 0; i < noNeighbours; i++){
            weights[i] = kernel(cosineDistance(neighbours[0], neighbours[i]));
        }
        return weights;
    }

    private double kernel(double distance){
        return Math.sqrt(Math.exp(-Math.pow(distance, 2) / Math.pow(0.25, 2))); //kernel width of 0.25 ^2
    }

    private double cosineDistance(double[] inst1, double[] inst2){
        return -(dot(inst1,inst2) / (Math.sqrt(dot(inst1,inst1)) * Math.sqrt(dot(inst2,inst2)))) + 1;
    }

    private double dot(double[] inst1, double[] inst2){
        double sum = 0;
        for (int i = 0; i < inst1.length; i++)
            sum += inst1[i] * inst2[i];
        return sum;
    }

    private double euclideanDistance(double[] inst1, double[] inst2){
        double distance = 0;
        for (int i = 0; i < inst1.length; i++){
            distance += Math.pow(inst1[i] - inst2[i], 2);
        }
        return Math.sqrt(distance);
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        //Minimum working example
        String dataset = "EthanolLevel";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset +
                "\\" + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset +
                "\\" + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        CIF c = new CIF();
        c.buildClassifier(train);
        LEFTIST l = new LEFTIST(train, c, 0);
        l.outputFigure(test.get(0), "E:\\Temp\\");
    }
}
