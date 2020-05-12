package tsml.classifiers.shapelet_based;

import java.io.File;
import java.io.FileReader;
import java.util.*;

import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.shapelet_tools.OrderLineObj;
import tsml.transformers.shapelet_tools.class_value.NormalClassValue;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.quality_measures.InformationGain;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;
import utilities.ClassifierTools;
import utilities.class_counts.ClassCounts;
import weka.core.*;

import tsml.transformers.shapelet_tools.Shapelet;


public class ShapeletTree extends EnhancedAbstractClassifier implements TechnicalInformationHandler{


    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "L Ye, E Keogh");
        result.setValue(TechnicalInformation.Field.TITLE, "Time series shapelets: a new primitive for data mining");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Proc. 15th SIGKDD");
        result.setValue(TechnicalInformation.Field.YEAR, "2009");
        return result;
    }


    public ShapeletQuality getQuality() {
        return quality;
    }

    public void setQuality(ShapeletQuality.ShapeletQualityChoice qualityChoice) {
        this.quality = new ShapeletQuality(qualityChoice);
    }

    public ShapeletDistance getSubseqDistance() {
        return subseqDistance;
    }

    public void setSubseqDistance(ShapeletDistance subseqDistance) {
        this.subseqDistance = subseqDistance;
    }

    public void setCandidatePruning(boolean f) {
        this.useCandidatePruning = f;
        this.candidatePruningStartPercentage = f ? 10 : 100;
    }

    protected int candidatePruningStartPercentage;
    protected boolean useCandidatePruning;


    protected Comparator<Shapelet> shapeletComparator = new Shapelet.LongOrder();

    //qualiyu/ distance and class value.
    protected ShapeletQuality quality;
    protected ShapeletDistance subseqDistance;
    protected NormalClassValue classValue;

    private ShapeletNode root;
    private String logFileName;
    private int minLength, maxLength;


    public ShapeletTree(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.root = new ShapeletNode();
        setQuality(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
        subseqDistance = new ShapeletDistance();
        classValue = new NormalClassValue();
    }

    public void setShapeletMinMaxLength(int minLength, int maxLength){
        this.minLength = minLength;
        this.maxLength = maxLength;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception{
        if(minLength < 1 || maxLength < 1){
            if(debug)
                System.out.println("Shapelet minimum or maximum length is incorrectly specified. Min = "+minLength+" max = "+maxLength+" setting to whole series");
            minLength=3;
            maxLength=data.numAttributes()-1;
        }
        long t1=System.nanoTime();
        root.initialiseNode(data, minLength, maxLength,0);
        trainResults.setBuildTime(System.nanoTime()-t1);
        trainResults.setParas(getParameters());
    }

    @Override
    public double classifyInstance(Instance instance) {
        return root.classifyInstance(instance);
    }

    private Shapelet getRootShapelet(){
        return this.root.shapelet;
    }

    /**
     *
     * @param classDist
     * @return
     */
    protected void initQualityBound(ClassCounts classDist) {
        if (!useCandidatePruning) return;
        quality.initQualityBound(classDist, candidatePruningStartPercentage);
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

        public void initialiseNode(Instances data, int minShapeletLength, int maxShapeletLength, int level) throws Exception{
            subseqDistance.init(data);
            classValue.init(data);

            if(debug)
                System.out.println(data.numInstances());

            // 1. check whether this is a leaf node with only one class present
            double firstClassValue = classValue.getClassValue(data.instance(0));
            boolean oneClass = true;
            for(int i = 1; i < data.numInstances(); i++){
                if(classValue.getClassValue(data.instance(i)) != firstClassValue){
                    oneClass = false;
                    break;
                }
            }

            if(oneClass){
                this.classDecision = firstClassValue; // no need to find shapelet, base case

            } else { // recursively call method to create left and right children nodes

                try{
                    // 1. find the best shapelet to split the data
                    this.shapelet = findBestShapelet(1,data,minShapeletLength, maxShapeletLength);


                    // 2. split the data using the shapelet and create new data sets
                    double dist;
                    ArrayList<Instance> splitLeft = new ArrayList<Instance>();
                    ArrayList<Instance> splitRight = new ArrayList<Instance>();

                    subseqDistance.setShapelet(shapelet); //set the shapelet for the distance function.
                    for(int i = 0; i < data.numInstances(); i++){
                        dist = subseqDistance.calculate(data.instance(i), i);

                        if(debug)
                            System.out.println(shapelet.splitThreshold + "  " + dist);
                        (dist < shapelet.splitThreshold ? splitLeft : splitRight).add(data.instance(i));
                    }

                    // 5. initialise and recursively compute children nodes
                    leftNode = new ShapeletNode();
                    rightNode = new ShapeletNode();
//                                System.out.println("SplitLeft:");

                    Instances leftInstances = new Instances(data, splitLeft.size());
                    leftInstances.addAll(splitLeft);
                    Instances rightInstances = new Instances(data, splitRight.size());
                    rightInstances.addAll(splitRight);

                    leftNode.initialiseNode(leftInstances, minShapeletLength, maxShapeletLength, (level+1));
//                                System.out.println("SplitRight:");
                    rightNode.initialiseNode(rightInstances, minShapeletLength, maxShapeletLength, (level+1));
                }catch(Exception e){
                    System.out.println("Problem initialising tree node: "+e);
                    e.printStackTrace();
                }
            }
        }

        public double classifyInstance(Instance instance){
            if (this.leftNode == null) {
                return this.classDecision;
            } else {
                double distance;
                subseqDistance.setShapelet(shapelet);
                distance = subseqDistance.calculate(instance, 0);

                if (distance < this.shapelet.splitThreshold) {
                    return leftNode.classifyInstance(instance);
                } else {
                    return rightNode.classifyInstance(instance);
                }
            }
        }

    }


    //#
    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {
        long startTime = System.nanoTime();
        this.findBestShapelet(1, data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double)(finishTime - startTime) / 1000000000.0;
    }

    // edited from findBestKShapeletsCached
    private Shapelet findBestShapelet(int numShapelets, Instances data, int minShapeletLength, int maxShapeletLength){
        ArrayList<Shapelet> kShapelets = new ArrayList<Shapelet>();         // store (upto) the best k shapelets overall
        ArrayList<Shapelet> seriesShapelets = new ArrayList<Shapelet>();    // temp store of all shapelets for each time series

        Shapelet bestShapelet = null;

        for(int i = 0; i < data.numInstances(); i++){
            subseqDistance.setSeries(i);
            double[] wholeCandidate = data.instance(i).toDoubleArray();

            for(int length = minShapeletLength; length <= maxShapeletLength; length++){
                for(int start = 0; start <= wholeCandidate.length - length-1; start++){ //-1 = avoid classVal - handle later for series with no class val
                    Shapelet candidateShapelet = checkCandidate(data, data.instance(i), i, start, length);
                    if(bestShapelet == null){
                        bestShapelet = candidateShapelet;
                    }

                    if(shapeletComparator.compare(bestShapelet, candidateShapelet) > 0){
                        bestShapelet = candidateShapelet;
                    }

                }
            }
        }
        if(debug)
            System.out.println("final.quality = " + bestShapelet.getQualityValue());

        return bestShapelet;
    }


    /**
     *
     * @param shapelets the input Shapelets to remove self similar Shapelet objects from
     * @return a copy of the input ArrayList with self-similar shapelets removed
     */
    private static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets){
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<Shapelet> outputShapelets = new ArrayList<Shapelet>();
        boolean[] selfSimilar = new boolean[shapelets.size()];

        // to keep tract of self similarity - assume nothing is similar to begin with
        for(int i = 0; i < shapelets.size(); i++){
            selfSimilar[i] = false;
        }

        for(int i = 0; i < shapelets.size();i++){
            if(!selfSimilar[i]){
                outputShapelets.add(shapelets.get(i));
                for(int j = i+1; j < shapelets.size(); j++){
                    if(!selfSimilar[i] && selfSimilarity(shapelets.get(i),shapelets.get(j))){ // no point recalc'ing if already self similar to something
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
    private ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets){

        ArrayList<Shapelet> newBestSoFar = new ArrayList<Shapelet>();
        kBestSoFar.addAll(timeSeriesShapelets);
        Collections.sort(kBestSoFar);
        if(kBestSoFar.size()<k)
            return kBestSoFar; // no need to return up to k, as there are not k shapelets yet

        for(int i = 0; i < k; i++){
            newBestSoFar.add(kBestSoFar.get(i));
        }

        return newBestSoFar;
    }


    protected Shapelet checkCandidate(Instances inputData, Instance series,int series_id, int start, int length) {
        //init qualityBound.
        initQualityBound(classValue.getClassDistributions());

        //set the candidate. This is the instance, start and length.
        subseqDistance.setCandidate(series, start, length, 0);

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        int dataSize = inputData.numInstances();

        for (int i = 0; i < dataSize; i++) {

            //Check if it is possible to prune the candidate
            if (quality.pruneCandidate()) {
                return null;
            }

            double distance = subseqDistance.calculate(inputData.instance(i), i);

            //this could be binarised or normal.
            double classVal = classValue.getClassValue(inputData.instance(i));

            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            quality.updateOrderLine(orderline.get(orderline.size() - 1));
        }

        Shapelet shapelet = new Shapelet(subseqDistance.getCandidate(), series_id, start, quality.getQualityMeasure());

        //this class distribution could be binarised or normal.
        shapelet.calculateQuality(orderline, classValue.getClassDistributions());
        shapelet.classValue = classValue.getShapeletValue(); //set classValue of shapelet. (interesing to know).

        //as per the way. We select our Shapelet and assess it's quality through the various methods. But we then calculate a splitting threshold with IG.
        shapelet.splitThreshold = InformationGain.calculateSplitThreshold(orderline, classValue.getClassDistributions());
        return shapelet;
    }

    private static double entropy(TreeMap<Double, Integer> classDistributions){
        if(classDistributions.size() == 1){
            return 0;
        }

        double thisPart;
        double toAdd;
        int total = 0;
        for(Double d : classDistributions.keySet()){
            total += classDistributions.get(d);
        }
        // to avoid NaN calculations, the individual parts of the entropy are calculated and summed.
        // i.e. if there is 0 of a class, then that part would calculate as NaN, but this can be caught and
        // set to 0.
        ArrayList<Double> entropyParts = new ArrayList<Double>();
        for(Double d : classDistributions.keySet()){
            thisPart =(double) classDistributions.get(d) / total;
            toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
            if(Double.isNaN(toAdd))
                toAdd=0;
            entropyParts.add(toAdd);
        }

        double entropy = 0;
        for(int i = 0; i < entropyParts.size(); i++){
            entropy += entropyParts.get(i);
        }
        return entropy;
    }

    /**
     *
     * @param fileName
     * @return
     */
    public static Instances loadData(String fileName){
        Instances data = null;
        try{
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        } catch(Exception e){
            System.out.println(" Error =" + e + " in method loadData");
        }
        return data;
    }

    private static boolean selfSimilarity(int seriesId, int startPos, int length, Shapelet[] selectedShapelets){
        boolean selfSimilarity = false;

        for(Shapelet shapelet : selectedShapelets){
            if(shapelet != null){
                if(seriesId == shapelet.seriesId){
                    if(startPos >= shapelet.startPos && startPos <= shapelet.startPos + shapelet.getLength()) //candidate starts within exisiting shapelet
                    {
                        selfSimilarity = true;
                    }
                    if(shapelet.startPos >= startPos && shapelet.startPos <= startPos + length){
                        selfSimilarity = true;
                    }
                }
            }
        }
        return selfSimilarity;
    }


    private static boolean selfSimilarity(Shapelet candidate, TreeSet<Shapelet> setOfShapelets){
        boolean selfSimilarity = false;
        for(Shapelet shapelet : setOfShapelets){
            if(shapelet != null){
                if(candidate.seriesId == shapelet.seriesId){
                    if(candidate.startPos >= shapelet.startPos && candidate.startPos <= shapelet.startPos + shapelet.getLength()) //candidate starts within exisiting shapelet
                    {
                        selfSimilarity = true;
                    }
                    if(shapelet.startPos >= candidate.startPos && shapelet.startPos <= candidate.startPos + candidate.getLength()){
                        selfSimilarity = true;
                    }
                }
            }
        }
        return selfSimilarity;
    }


    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate){
        if(candidate.seriesId == shapelet.seriesId){
            if(candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.getLength()){ //candidate starts within exisiting shapelet
                return true;
            }
            if(shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.getLength()){
                return true;
            }
        }
        return false;
    }


    public static String getTime(){
        Calendar calendar = new GregorianCalendar();
        return calendar.get(Calendar.DAY_OF_MONTH)+"/"+calendar.get(Calendar.MONTH)+"/"+calendar.get(Calendar.YEAR)+" - "+calendar.get(Calendar.HOUR_OF_DAY)+":"+calendar.get(Calendar.MINUTE)+":"+calendar.get(Calendar.SECOND)+" AM";
    }


    public static void main(String[] args) throws Exception {

        final String resampleLocation = "D:\\Research TSC\\Data\\TSCProblems2018";
        final String dataset = "ItalyPowerDemand";
        final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;
        System.out.println(filePath);
        Instances test, train;
        test = DatasetLoading.loadDataNullable(filePath + "_TEST");
        train = DatasetLoading.loadDataNullable(filePath + "_TRAIN");

        ShapeletTree stc = new ShapeletTree();
        stc.setShapeletMinMaxLength(3, train.numAttributes()-1);
        stc.buildClassifier(train);

        double accuracy = ClassifierTools.accuracy(test, stc);

        System.out.println("ShapeletTree accuracy uis: " + accuracy);
    }

}

