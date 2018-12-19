package timeseriesweka.filters.shapelet_transforms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality.ShapeletQualityChoice;
import weka.filters.unsupervised.instance.Resample;


/**
 * An approximate filter to transform a dataset by k shapelets. The approximation
 * is achieved by means of sampling the dataset according to supplied percentages
 * 
 * @author Edgaras Baranauskas
 */

@Deprecated
public class ApproximateShapeletTransform extends ShapeletTransform{
    /**
     * Size of the subsample, as a percentage of the original set 
     */
    protected int seriesSampleLevel;
    
    /**
     * Size of approximated series, as a percentage of the original series
     */
    protected int dataPointsSize;  
    
    private ArrayList<Integer> sampledIDs;
    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ApproximateShapeletTransform(){
        super();
        seriesSampleLevel = 50;
        dataPointsSize = 50;
    }

    /**
     * Single param constructor: filter is unusable until min/max params are initialised.
     * Quality measure defaults to information gain.
     * @param k the number of shapelets to be generated
     */
    public ApproximateShapeletTransform(int k){
        super(k);
        seriesSampleLevel = 50;
        dataPointsSize = 50;
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ApproximateShapeletTransform(int k, int minShapeletLength, int maxShapeletLength){
        super(k, minShapeletLength, maxShapeletLength);
        seriesSampleLevel = 50;
        dataPointsSize = 50;
    }

    /**
     * Full, exhaustive, constructor for a filter. Quality measure set via enum, invalid 
     * selection defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     * @param qualityChoice the shapelet quality measure to be used with this filter
     */
    public ApproximateShapeletTransform(int k, int minShapeletLength, int maxShapeletLength, ShapeletQualityChoice qualityChoice){
        super(k, minShapeletLength, maxShapeletLength, qualityChoice);
        seriesSampleLevel = 50;
        dataPointsSize = 50;
    }
    
    /**
     * Method to set the sampling levels for series and data points. The default
     * percentages are 50, 50.
     * 
     * @param series the percentage of series to be sampled
     * @param dataPoints the percentage of data points to be used in PAA series
     */
    public void setSampleLevels(int series, int dataPoints) throws IOException{
        if(series < 1 || series > 100){
            throw new IOException ("Series sample level must be in range [1, 100]");
        }
                
        if(dataPoints < 1 || dataPoints > 100){
            throw new IOException ("Piece aggregate approximation must be in range [1, 100]");
        } 
                
        seriesSampleLevel = series;
        dataPointsSize = dataPoints;
    }
    
    
    @Override
    public Instances process(Instances dataInst) throws IllegalArgumentException
    {

        //check the input data is correct and assess whether the filter has been setup correctly.
        inputCheck(dataInst);   


        //Approximate data
        Instances orderedInst = null;
        if(!this.m_FirstBatchDone){
            sampledIDs = new ArrayList<Integer>();
            dataInst = approximateInstanes(dataInst);
            
            //Sort data in round robin order
            dataSourceIDs = new int[dataInst.numInstances()];
            int[] roundRobidIDs = new int[dataInst.numInstances()];
            orderedInst = roundRobinData(dataInst, roundRobidIDs);
            
            //Generate ID of the orignal source
            dataSourceIDs = new int[dataInst.numInstances()];
            for(int i = 0; i < dataSourceIDs.length; i++){
                dataSourceIDs[i] = sampledIDs.get(roundRobidIDs[i]);
            }
            
        }else{
            dataInst = performPAA(dataInst);
        }
            
        if(!m_FirstBatchDone){ // shapelets discovery has not yet been caried out, so do so
            this.shapelets = findBestKShapeletsCache(orderedInst); // get k shapelets ATTENTION
            m_FirstBatchDone = true;
            if(!supressOutput){
                System.out.println(shapelets.size()+" Shapelets have been generated");
            }
        }

        return this.buildTansformedDataset(dataInst);
    }
    
    //Method to apprimiate the training data
    private Instances approximateInstanes(Instances data){
        Instances output = sampleInstances(data);
        output = performPAA(output);
        
        //Make shapelet length relative to that of the original
        //minShapeletLength = (output.numAttributes() - 1) * minShapeletLength / (data.numAttributes()-1);
        //maxShapeletLength = (output.numAttributes() - 1) * maxShapeletLength / (data.numAttributes()-1);
        
        return output;
    }
    
    //Method to sample instances
    private Instances sampleInstances(Instances data){
        if(seriesSampleLevel == 100){
            return data;
        }else{
            Resample sampler = new Resample();

            //Set up sampler
            try {
                sampler.setInputFormat(data);
            } catch (Exception ex) {
                Logger.getLogger(ApproximateShapeletTransform.class.getName()).log(Level.SEVERE, null, ex);
            }

            sampler.setNoReplacement(true);
            sampler.setSampleSizePercent(seriesSampleLevel);

            //Queue data for processing
            for(int i = 0; i < data.numInstances(); i++){
                sampler.input(data.instance(i));
            }
            sampler.batchFinished();

            //Retrieve output
            Instances sampledData = new Instances(data, data.numInstances() * seriesSampleLevel / 100);
            boolean isFinished = false;
            while(!isFinished){
                Instance toAdd = sampler.output();
                if(toAdd == null){
                    isFinished = true;
                }else{
                    sampledData.add(toAdd);
                    
                    //Find source id
                    for(int sIndex = 0; sIndex < data.numInstances(); sIndex++){
                        for(int attIndex = 0; attIndex < data.numAttributes(); attIndex++){
                            if(data.instance(sIndex).value(attIndex) != toAdd.value(attIndex)){
                                break;
                            }else if(attIndex == data.numAttributes()-1){
                                sampledIDs.add(sIndex);
                            }
                        }
                    }
                   
                }
            }

            /* Used for testing
            TreeMap<Double, Integer> dist = FullShapeletTransform.getClassDistributions(data);
            TreeMap<Double, Integer> dist2 = FullShapeletTransform.getClassDistributions(sampledData);
            printTreeMap(dist);
            printTreeMap(dist2);

            System.out.println("Original size: " + data.numInstances());
            System.out.println("Percentage: " + seriesSampleLevel);
            System.out.println("Sampled size: " + sampledData.numInstances());
            */

            return sampledData;
        }
    }
    
    //Method to perform Piecewise Aggregate Approximation for a given data
    private Instances performPAA(Instances data){
                
        if(dataPointsSize == 100){
            return data;
        }else{ 
            int paaSize = (data.numAttributes()-1) * dataPointsSize / 100;
            //Determine output format
            Instances output = null;
            try {
                output = determinePAAOutputFormat(data, paaSize);
            } catch (Exception ex) {
                Logger.getLogger(ApproximateShapeletTransform.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            double portionLength = ((double)(data.numAttributes() - 1)) / paaSize;
            
            //For each data, compute PAA components
            for(int i = 0; i < data.numInstances(); i++){ 
                Instance currentInstance = data.instance(i);
                Instance toAdd = new DenseInstance(paaSize + 1);
                
                //Normalise series
                double[] series = currentInstance.toDoubleArray();
                series = this.subseqDistance.zNormalise(series, true);
                
                double[] paaSublists = new double[paaSize];
                int[] paaSublistsSizes = new int[paaSize];
                
                double currentPortion = portionLength;
                int seriesIndex = 0;
                int subListIndex = 0;
                
                boolean advance = false;
                while(!advance){
                    if(currentPortion >= 0.999999999999){//Get rid of accumulated error
                        paaSublistsSizes[subListIndex]++;
                        paaSublists[subListIndex] += series[seriesIndex++];
                        currentPortion -= 1.0;
                        if(currentPortion < 0.0){
                            currentPortion = 0.0;
                        }
                    }else{
                        if(seriesIndex < series.length-1){
                            //Required portion
                            paaSublistsSizes[subListIndex]++;
                            paaSublists[subListIndex++] += currentPortion * series[seriesIndex];
                            
                            //Remaining portion
                            currentPortion = 1.0 - currentPortion;
                            paaSublistsSizes[subListIndex]++;
                            paaSublists[subListIndex] += currentPortion * series[seriesIndex];
                            currentPortion = portionLength - currentPortion;
                            
                        }else{
                            advance = true;
                        }
                        seriesIndex++;  
                    }
                }
                
                for(int j = 0; j < paaSublists.length; j++){
                    toAdd.setValue(j, paaSublists[j]/paaSublistsSizes[j]);
                }
                toAdd.setValue(paaSize, currentInstance.classValue());    
                output.add(toAdd);
            }
            return output;
        }
    }
    
    //Method to determine output format of Piecewise Aggregate Approximation of the time series
    private Instances determinePAAOutputFormat(Instances inputFormat, int length) throws Exception{

        FastVector atts = new FastVector();
        String name;
        for(int i = 0; i < length; i++){
            name = "PAA" + i;
            atts.addElement(new Attribute(name));
        }

        if(inputFormat.classIndex() >= 0){ //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for(int i = 0; i < target.numValues(); i++){
                vals.addElement(target.value(i));
            }
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("PAA" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if(inputFormat.classIndex() >= 0){
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }
    
    //Method used for testing
    private void printTreeMap(TreeMap<Double, Integer> dist){
        System.out.println("\nTREEMAP");
        for(Double d: dist.keySet()){
            System.out.println(d + ": " +dist.get(d));
        }
    }
    
    //Method used for testing
    private double[] testPAA(double[] data) throws IOException{
        FastVector atts = new FastVector();
        String name;
        for(int i = 0; i < data.length-1; i++){
            name = "Attribute" + i;
            atts.addElement(new Attribute(name));
        }

        FastVector classValues = new FastVector();
        classValues.addElement("0");
        classValues.addElement("1");
        Attribute classAtt = new Attribute("Binary", classValues);
        atts.addElement(classAtt);
        
        //Create dataset
        Instances instances = new Instances("Test", atts, 1);
        instances.setClassIndex(data.length-1);

        //Create instance
        Instance inst = new DenseInstance(1, data);
        instances.add(inst);

        
        Instances output = performPAA(instances);
        
        return output.instance(0).toDoubleArray();
    }
  
    /**
     *
     * @param args
     */
    public static void main(String[] args){        
        //Create some time series for testing
        System.out.println("\n1.) Create series for testing: ");
        int seriesLength = 11;
        
        double[] dataEven = new double[seriesLength];
        
        int min = -5;
        int max = 5;
        for(int j = 0; j < seriesLength; j++){
            if(j == seriesLength-1){
                dataEven[j] = 0;
            }else{
                dataEven[j] = min + (int)(Math.random() * ((max - min) + 1));
            }
        }
        
        seriesLength = 10;
        double[] dataUneven = new double[seriesLength];
        for(int j = 0; j < seriesLength; j++){
            if(j == seriesLength-1){
                dataUneven[j] = 0;
            }else{
                dataUneven[j] = min + (int)(Math.random() * ((max - min) + 1));
            }
        }
        
        ApproximateShapeletTransform ast = new ApproximateShapeletTransform();
        
        double[] out = null;
        try {
            ast.setSampleLevels(100, 50);
            out = ast.testPAA(dataEven);
        } catch (IOException ex) {
            Logger.getLogger(ApproximateShapeletTransform.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        System.out.println("Even Test: ");
    }
    
}
