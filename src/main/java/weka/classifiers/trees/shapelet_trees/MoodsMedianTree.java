/*
     * copyright: Anthony Bagnall
 * */package weka.classifiers.trees.shapelet_trees;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeMap;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

public class MoodsMedianTree extends AbstractClassifier{

    private ShapeletNode root;
    private String logFileName;
    private int minLength, maxLength;
       
    public MoodsMedianTree(String logFileName) throws Exception {
        this.root = new ShapeletNode();
        this.logFileName = logFileName;
        FileWriter fw = new FileWriter(logFileName);
        fw.close();
    }

    public void setShapeletMinMaxLength(int minLength, int maxLength){
        this.minLength = minLength;
        this.maxLength = maxLength;
    }
        
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(minLength < 1 || maxLength < 1){
            throw new Exception("Shapelet minimum or maximum length is incorrectly specified!");
        }
        
        root.initialiseNode(data, minLength, maxLength, 0);
    }

    @Override
    public double classifyInstance(Instance instance) {
        return root.classifyInstance(instance);
    }

    private Shapelet getRootShapelet() {
        return this.root.shapelet;
    }

    private class ShapeletNode {

        private ShapeletNode leftNode;
        private ShapeletNode rightNode;
        private double classDecision;
        private Shapelet shapelet;

        public ShapeletNode() {
            leftNode = null;
            rightNode = null;
            classDecision = -1;
        }

        public void initialiseNode(Instances data, int minShapeletLength, int maxShapeletLength, int level) throws Exception {
            FileWriter fw = new FileWriter(logFileName, true);
            fw.append("level:" + level + ", numInstances:" + data.numInstances() + "\n");
            fw.close();

            // 1. check whether this is a leaf node with only one class present
            double firstClassValue = data.instance(0).classValue();
            boolean oneClass = true;
            for (int i = 1; i < data.numInstances(); i++) {
                if (data.instance(i).classValue() != firstClassValue) {
                    oneClass = false;
                    break;
                }
            }

            if (oneClass == true) {
                this.classDecision = firstClassValue; // no need to find shapelet, base case
//                                System.out.println("base case");
                fw = new FileWriter(logFileName, true);
                fw.append("class decision here: " + firstClassValue + "\n");
                fw.close();
            } else { // recursively call method to create left and right children nodes
                try {
                    // 1. find the best shapelet to split the data
                    this.shapelet = findBestShapelet(data, minShapeletLength, maxShapeletLength);

                    // 2. split the data using the shapelet and create new data sets
                    double dist;
//                                System.out.println("Threshold:"+shapelet.getThreshold());
//                                System.out.println("length:"+shapelet.getLength());
                    ArrayList<Instance> splitLeft = new ArrayList<Instance>();
                    ArrayList<Instance> splitRight = new ArrayList<Instance>();

                    for (int i = 0; i < data.numInstances(); i++) {
                        dist = subsequenceDistance(this.shapelet.content, data.instance(i).toDoubleArray());
//                                System.out.println("dist:"+dist);
                        if (dist < shapelet.medianDistance) {
                            splitLeft.add(data.instance(i));
//                                                System.out.println("gone left");
                        } else {
                            splitRight.add(data.instance(i));
//                                                System.out.println("gone right");
                        }
                    }

                    // write to file here!!!!
                    fw = new FileWriter(logFileName, true);
                    fw.append("seriesId, startPos, length, infoGain, splitThresh\n");
                    fw.append(this.shapelet.seriesId + "," + this.shapelet.startPos + "," + this.shapelet.content.length + "," + this.shapelet.moodsMedianStat + "," + this.shapelet.medianDistance + "\n");
                    for (int j = 0; j < this.shapelet.content.length; j++) {
                        fw.append(this.shapelet.content[j] + ",");
                    }
                    fw.append("\n");
                    fw.close();

                    System.out.println("shapelet completed at:" + System.nanoTime());


//                        System.out.println("leftSize:"+splitLeft.size());
//                        System.out.println("leftRight:"+splitRight.size());

                    // 5. initialise and recursively compute children nodes
                    leftNode = new ShapeletNode();
                    rightNode = new ShapeletNode();
//                                System.out.println("SplitLeft:");

                    Instances leftInstances = new Instances(data, splitLeft.size());
                    for (int i = 0; i < splitLeft.size(); i++) {
                        leftInstances.add(splitLeft.get(i));
                    }
                    Instances rightInstances = new Instances(data, splitRight.size());
                    for (int i = 0; i < splitRight.size(); i++) {
                        rightInstances.add(splitRight.get(i));
                    }

                    fw = new FileWriter(logFileName, true);
                    fw.append("left size under level " + level + ": " + leftInstances.numInstances() + "\n");
                    fw.close();
                    leftNode.initialiseNode(leftInstances, minShapeletLength, maxShapeletLength, (level + 1));
//                                System.out.println("SplitRight:");

                    fw = new FileWriter(logFileName, true);
                    fw.append("right size under level " + level + ": " + rightInstances.numInstances() + "\n");
                    fw.close();

                    rightNode.initialiseNode(rightInstances, minShapeletLength, maxShapeletLength, (level + 1));
                } catch (Exception e) {
                    System.out.println("Problem initialising tree node: " + e);
                    e.printStackTrace();
                }
            }
        }

        public double classifyInstance(Instance instance) {
            if (this.leftNode == null) {
                return this.classDecision;
            } else {
                double distance;
                distance = subsequenceDistance(this.shapelet.content, instance);

                if (distance < this.shapelet.medianDistance) {
                    return leftNode.classifyInstance(instance);
                } else {
                    return rightNode.classifyInstance(instance);
                }
            }
        }
    }

    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {
        long startTime = System.nanoTime();
        this.findBestShapelet(data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double)(finishTime - startTime) / 1000000000.0;
    }

    //#
    // edited from findBestKShapeletsCached
    private Shapelet findBestShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {

        Shapelet bestShapelet = null;


        TreeMap<Double, Integer> classDistributions = getClassDistributions(data); // used to calc info gain

        //for all time series
        System.out.println("Processing data: ");
        for (int i = 0; i < data.numInstances(); i++) {
//                System.out.println((1+i)+"/"+data.numInstances()+"\t Started: "+getTime());

            double[] wholeCandidate = data.instance(i).toDoubleArray();
            // for all lengths
            for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
                //for all possible starting positions of that length
                for (int start = 0; start <= wholeCandidate.length - length - 1; start++) { //-1 = avoid classVal - handle later for series with no class val
                    // CANDIDATE ESTABLISHED - got original series, length and starting position
                    // extract relevant part into a double[] for processing
                    double[] candidate = new double[length];
                    for (int m = start; m < start + length; m++) {
                        candidate[m - start] = wholeCandidate[m];
                    }

                    candidate = zNorm(candidate, false);
                    Shapelet candidateShapelet = checkCandidate(candidate, data, i, start, classDistributions);

                    if (bestShapelet == null || candidateShapelet.compareTo(bestShapelet) < 0) {
                        bestShapelet = candidateShapelet;
                    }

                }
            }
        }

        // note - no need to calculate the split dist, median point, etc as this is all inferred from the unordered calcs in checkCandidate()

        //print out the k best shapes and then return
//        System.out.println("Shapelet No, Series ID, Start, Length, InfogGain, Gap,");

        return bestShapelet;
    }

    /**
     *
     * @param shapelets the input Shapelets to remove self similar Shapelet objects from
     * @return a copy of the input ArrayList with self-similar shapelets removed
     */
    private static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets) {
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<Shapelet> outputShapelets = new ArrayList<Shapelet>();
        boolean[] selfSimilar = new boolean[shapelets.size()];

        // to keep tract of self similarity - assume nothing is similar to begin with
        for (int i = 0; i < shapelets.size(); i++) {
            selfSimilar[i] = false;
        }

        for (int i = 0; i < shapelets.size(); i++) {
            if (selfSimilar[i] == false) {
                outputShapelets.add(shapelets.get(i));
                for (int j = i + 1; j < shapelets.size(); j++) {
                    if (selfSimilar[j] == false && selfSimilarity(shapelets.get(i), shapelets.get(j))) { // no point recalc'ing if already self similar to something
                        selfSimilar[j] = true;
                    }
                }
            }
        }
        return outputShapelets;
    }

    /**
     *
     * @param k the maximum number of shapelets to be returned after combining the two lists
     * @param kBestSoFar the (up to) k best shapelets that have been observed so far, passed in to combine with shapelets from a new series
     * @param timeSeriesShapelets the shapelets taken from a new series that are to be merged in descending order of fitness with the kBestSoFar
     * @return an ordered ArrayList of the best k (or less) Shapelet objects from the union of the input ArrayLists
     */
    private ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets) {

        ArrayList<Shapelet> newBestSoFar = new ArrayList<Shapelet>();
        for (int i = 0; i < timeSeriesShapelets.size(); i++) {
            kBestSoFar.add(timeSeriesShapelets.get(i));
        }
        Collections.sort(kBestSoFar);
        if (kBestSoFar.size() < k) {
            return kBestSoFar; // no need to return up to k, as there are not k shapelets yet
        }
        for (int i = 0; i < k; i++) {
            newBestSoFar.add(kBestSoFar.get(i));
        }

        return newBestSoFar;
    }

    /**
     *
     * @param data the input data set that the class distributions are to be derived from
     * @return a TreeMap<Double, Integer> in the form of <Class Value, Frequency>
     */
    private static TreeMap<Double, Integer> getClassDistributions(Instances data) {
        TreeMap<Double, Integer> classDistribution = new TreeMap<Double, Integer>();
        double classValue;
        for (int i = 0; i < data.numInstances(); i++) {
            classValue = data.instance(i).classValue();
            boolean classExists = false;
            for (Double d : classDistribution.keySet()) {
                if (d == classValue) {
                    int temp = classDistribution.get(d);
                    temp++;
                    classDistribution.put(classValue, temp);
                    classExists = true;
                }
            }
            if (classExists == false) {
                classDistribution.put(classValue, 1);
            }
        }
        return classDistribution;
    }

    /**
     *
     * @param candidate the data from the candidate Shapelet
     * @param data the entire data set to compare the candidate to
     * @param data the entire data set to compare the candidate to
     * @return a TreeMap<Double, Integer> in the form of <Class Value, Frequency>
     */
    private static Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, TreeMap classDistribution) {

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<OrderLineObj>();

        for (int i = 0; i < data.numInstances(); i++) {
            double distance = subsequenceDistance(candidate, data.instance(i));
//            System.out.println("dist: "+distance);
            double classVal = data.instance(i).classValue();

//            boolean added = false;
            // add to orderline
//            if(orderline.isEmpty()){
//                orderline.add(new OrderLineObj(distance, classVal));
//                added = true;
//            } else{
//                for(int j = 0; j < orderline.size(); j++){
//                    if(added == false && orderline.get(j).distance > distance){
//                        orderline.add(j, new OrderLineObj(distance, classVal));
//                        added = true;
//                    }
//                }
//            }
//            // if obj hasn't been added, must be furthest so add at end
//            if(added == false){
//                orderline.add(new OrderLineObj(distance, classVal));
//            }

            // CHANGED HERE! No need for orderline to be ordered..
            orderline.add(new OrderLineObj(distance, classVal));


        }

        Shapelet shapelet = new Shapelet(candidate, seriesId, startPos);
        shapelet.calculateMoodsMedian(orderline, classDistribution);

        return shapelet;
    }

    /**
     *
     * @param candidate
     * @param timeSeriesIns
     * @return
     */
    public static double subsequenceDistance(double[] candidate, Instance timeSeriesIns) {
        double[] timeSeries = timeSeriesIns.toDoubleArray();
        return subsequenceDistance(candidate, timeSeries);
    }

    public static double subsequenceDistance(double[] candidate, double[] timeSeries) {

//        double[] timeSeries = timeSeriesIns.toDoubleArray();
        double bestSum = Double.MAX_VALUE;
        double sum = 0;
        double[] subseq;

        // for all possible subsequences of two
        for (int i = 0; i <= timeSeries.length - candidate.length - 1; i++) {
            sum = 0;
            // get subsequence of two that is the same lenght as one
            subseq = new double[candidate.length];

            for (int j = i; j < i + candidate.length; j++) {
                subseq[j - i] = timeSeries[j];
            }
            subseq = zNorm(subseq, false); // Z-NORM HERE
            for (int j = 0; j < candidate.length; j++) {

                sum += (candidate[j] - subseq[j]) * (candidate[j] - subseq[j]);

            }
            if (sum < bestSum) {
                bestSum = sum;
            }
        }
        return (1.0 / candidate.length * bestSum);
    }

    /**
     *
     * @param input
     * @param classValOn
     * @return
     */
    public static double[] zNorm(double[] input, boolean classValOn) {
        double mean;
        double stdv;

        double classValPenalty = 0;
        if (classValOn) {
            classValPenalty = 1;
        }
        double[] output = new double[input.length];
        double seriesTotal = 0;

        for (int i = 0; i < input.length - classValPenalty; i++) {
            seriesTotal += input[i];
        }

        mean = seriesTotal / (input.length - classValPenalty);
        stdv = 0;
        for (int i = 0; i < input.length - classValPenalty; i++) {
            stdv += (input[i] - mean) * (input[i] - mean);
        }

        stdv = stdv / input.length - classValPenalty;
        stdv = Math.sqrt(stdv);
//        System.out.println("stdv: "+stdv);

        if (stdv == 0) { // will cause a NaN error otherwise! If sdev == 0, data must all be the same value i.e. a straight line, so set all to 0
            for (int i = 0; i < input.length - classValPenalty; i++) {
                output[i] = 0;
            }
        } else { // standard case
            for (int i = 0; i < input.length - classValPenalty; i++) {
                output[i] = (input[i] - mean) / stdv;
            }
        }

        if (classValOn == true) {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    /**
     *
     * @param fileName
     * @return
     */
    public static Instances loadData(String fileName) {
        Instances data = null;
        try {
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println(" Error =" + e + " in method loadData");
        }
        return data;
    }

    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate) {
        if (candidate.seriesId == shapelet.seriesId) {
            if (candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.content.length) { //candidate starts within exisiting shapelet
                return true;
            }
            if (shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.content.length) {
                return true;
            }
        }
        return false;
    }

    private static class Shapelet implements Comparable<Shapelet> {

        private double[] content;
        private int seriesId;
        private int startPos;
        private double moodsMedianStat;
        private double medianDistance;

        private Shapelet(double[] content, int seriesId, int startPos) {
            this.content = content;
            this.seriesId = seriesId;
            this.startPos = startPos;
        }

        private Shapelet(double[] content) {
            this.content = content;
        }

        public void calculateMoodsMedian(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistributions) {

            Collections.sort(orderline);
            int lengthOfOrderline = orderline.size();
            double median;
            if (lengthOfOrderline % 2 == 0) {
                median = (orderline.get(lengthOfOrderline / 2).distance + orderline.get(lengthOfOrderline / 2 - 1).distance) / 2;
            } else {
                median = orderline.get(lengthOfOrderline / 2).distance;
            }

            TreeMap<Double, Integer> classCountsBelowMedian = new TreeMap<Double, Integer>();
            TreeMap<Double, Integer> classCountsAboveMedian = new TreeMap<Double, Integer>();

            for (Double d : classDistributions.keySet()) {
                classCountsBelowMedian.put(d, 0);
                classCountsAboveMedian.put(d, 0);
            }

            int totalCount = orderline.size();
            int countBelow = 0;
            int countAbove = 0;

            double distance;
            double classVal;
            int countSoFar;
            // count class distributions above and below the median
            for (int i = 0; i < orderline.size(); i++) {
                distance = orderline.get(i).distance;
                classVal = orderline.get(i).classVal;
                if (distance < median) {
                    countBelow++;
                    countSoFar = classCountsBelowMedian.get(classVal);
                    classCountsBelowMedian.put(classVal, countSoFar + 1);
                } else {
                    countAbove++;
                    countSoFar = classCountsAboveMedian.get(classVal);
                    classCountsAboveMedian.put(classVal, countSoFar + 1);
                }
            }

            double chi = 0;
            double expectedAbove, expectedBelow;
            for (Double d : classDistributions.keySet()) {

                expectedBelow = (double) (countBelow * classDistributions.get(d)) / totalCount;
                chi += ((classCountsBelowMedian.get(d) - expectedBelow) * (classCountsBelowMedian.get(d) - expectedBelow)) / expectedBelow;

                expectedAbove = (double) (countAbove * classDistributions.get(d)) / totalCount;
                chi += ((classCountsAboveMedian.get(d) - expectedAbove)) * (classCountsAboveMedian.get(d) - expectedAbove) / expectedAbove;
            }

            if (Double.isNaN(chi)) {
                chi = 0; // fix for cases where the shapelet is a straight line and chi is calc'd as NaN
            }

            this.moodsMedianStat = chi;
            this.medianDistance = median;
//                System.out.println("chi2: "+chi);

        }

        public double getMoodsMedianStat() {
            return this.moodsMedianStat;
        }

        public int getLength() {
            return this.content.length;
        }

        // comparison 1: to determine order of shapelets in terms of info gain, then separation gap, then shortness
        public int compareTo(Shapelet shapelet) {
            final int BEFORE = -1;
            final int EQUAL = 0;
            final int AFTER = 1;

            if (this.moodsMedianStat != shapelet.getMoodsMedianStat()) {
                if (this.moodsMedianStat > shapelet.getMoodsMedianStat()) {
                    return BEFORE;
                } else {
                    return AFTER;
                }
            } else if (this.content.length != shapelet.getLength()) {
                if (this.content.length < shapelet.getLength()) {
                    return BEFORE;
                } else {
                    return AFTER;
                }
            } else {
                return EQUAL;
            }
        }
    }

    private static class OrderLineObj implements Comparable<OrderLineObj> {

        private double distance;
        private double classVal;
        private double rank;

        private OrderLineObj(double distance, double classVal) {
            this.distance = distance;
            this.classVal = classVal;
            this.rank = -1;
        }

        public int compareTo(OrderLineObj o) {
            if (this.distance < o.distance) {
                return -1;
            } else if (this.distance == o.distance) {
                return 0;
            } else {
                return 1;
            }
        }
    }

//    private static double[] orderlineToDoubleArray(ArrayList<OrderLineObj> orderline) {
//        double output[] = new double[orderline.size()];
//        for(int i = 0; i < orderline.size(); i++){
//            output[i] = orderline.get(i).distance;
//        }
//        return output;
//    }
}
