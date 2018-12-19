/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms;

import development.DataSets;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;

/**
 *
 * @author Aaron
 */
public class GraceShapeletTransform extends ShapeletTransform {

    int currentSeries = 0;

    String seriesShapeletsFilePath;

    public void setSeries(int i) {
        currentSeries = i;
    }

    /**
     * The main logic of the filter; when called for the first time, k shapelets
     * are extracted from the input Instances 'data'. The input 'data' is
     * transformed by the k shapelets, and the filtered data is returned as an
     * output.
     * <p>
     * If called multiple times, shapelet extraction DOES NOT take place again;
     * once k shapelets are established from the initial call to process(), the
     * k shapelets are used to transform subsequent Instances.
     * <p>
     * Intended use:
     * <p>
     * 1. Extract k shapelets from raw training data to build filter;
     * <p>
     * 2. Use the filter to transform the raw training data into transformed
     * training data;
     * <p>
     * 3. Use the filter to transform the raw testing data into transformed
     * testing data (e.g. filter never extracts shapelets from training data,
     * therefore avoiding bias);
     * <p>
     * 4. Build a classifier using transformed training data, perform
     * classification on transformed test data.
     *
     * @param data the input data to be transformed (and to find the shapelets
     * if this is the first run)
     * @return the transformed representation of data, according to the
     * distances from each instance to each of the k shapelets
     */
    @Override
    public Instances process(Instances data) throws IllegalArgumentException {
        //check the input data is correct and assess whether the filter has been setup correctly.
        inputCheck(data);

        //setup classsValue
        classValue.init(data);
        //setup subseqDistance
        subseqDistance.init(data);

        //checks if the shapelets haven't been found yet, finds them if it needs too.
        if (!m_FirstBatchDone) {
            trainShapelets(data);
            m_FirstBatchDone = false; //set the shapelets Trained to false, because we'll set it to true once all the sub code has been finished.
            outputPrint("Partially Built the shapelet Set");
            return null;
        }

        //build the transformed dataset with the shapelets we've found either on this data, or the previous training data
        return buildTansformedDataset(data);
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data) {
        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series

        int proportion = numShapelets/data.numClasses();
        
        //for all time series
        outputPrint("Processing data: ");

        outputPrint("data : " + currentSeries);
        //we don't have a worst shapelet because we're doing a single scan.
        
        //set the series we're working with.
        subseqDistance.setSeries(currentSeries);
        //set the clas value of the series we're working with.
        classValue.setShapeletValue(data.get(currentSeries));

        seriesShapelets = searchFunction.SearchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
        
        Collections.sort(seriesShapelets, shapeletComparator);

        seriesShapelets = removeSelfSimilar(seriesShapelets);

        //by putting them into kShapelets we cut down on how many we seralise.
        //also use the proportion rather than num to be in line with Balanced.
        kShapelets = combine(proportion, kShapelets, seriesShapelets);

        createSerialFile(kShapelets);

        return kShapelets;
    }
    
    
    private void createSerialFile(ArrayList<Shapelet> shapelets){
        
        String fileName = getSubShapeletFileName(currentSeries);
        
        //Serialise the object.
        ObjectOutputStream out = null;
        try {
            out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(shapelets);
        } catch (IOException ex) {
            System.out.println("Failed to write " + ex);
        }
        finally{
            if(out != null){
                try {
                    out.close();
                } catch (IOException ex) {
                    System.out.println("Failed to close " + ex);
                }
            }
        }
    }
    
    private String getSubShapeletFileName(int i)
    {
        File f = new File(serialName);
        String str = f.getName();
        str = str.substring(0, str.lastIndexOf('.'));
        return str + "_" + i + ".ser";
    }

    //we use the balanced class structure from BalancedClassShapeletTransform.
    public Instances processFromSubFile(Instances train) {
        File f = new File(this.ouputFileLocation);

        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> seriesShapelets;
        
        TreeMap<Double, ArrayList<Shapelet>> kShapeletsMap = new TreeMap<>();
        for (int i=0; i < train.numClasses(); i++){
            kShapeletsMap.put((double)i, new ArrayList<Shapelet>());
        }
            
        //found out how many we want in each sub list.
        int proportion = numShapelets/kShapeletsMap.keySet().size();
        
        
        for(int i=0; i<train.numInstances(); i++){
            //get the proportion.
            kShapelets = kShapeletsMap.get(train.get(i).classValue());
            
            seriesShapelets = readShapeletsFromFile(getSubShapeletFileName(i));
            kShapelets = combine(proportion, kShapelets, seriesShapelets);
            
            //put the new proportion back.
            kShapeletsMap.put(train.get(i).classValue(), kShapelets);
        }
        
        kShapelets = buildKShapeletsFromMap(kShapeletsMap);
        
        this.numShapelets = kShapelets.size();
        
        shapelets = kShapelets;
        m_FirstBatchDone = true;

        return buildTansformedDataset(train);
    }
    
           
    private ArrayList<Shapelet> buildKShapeletsFromMap(Map<Double, ArrayList<Shapelet>> kShapeletsMap)
    {
       ArrayList<Shapelet> kShapelets = new ArrayList<>();
       
       int numberOfClassVals = kShapeletsMap.keySet().size();
       int proportion = numShapelets/numberOfClassVals;
       
       
       Iterator<Shapelet> it;
       
       //all lists should be sorted.
       //go through the map and get the sub portion of best shapelets for the final list.
       for(ArrayList<Shapelet> list : kShapeletsMap.values())
       {
           int i=0;
           it = list.iterator();
           
           while(it.hasNext() && i++ <= proportion)
           {
               kShapelets.add(it.next());
           }
       }
       return kShapelets;
    }
    
    

    public static ArrayList<Shapelet> readShapeletsFromFile(String shapeletLocation){
        ArrayList<Shapelet> shapelets = null;
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(shapeletLocation));
            shapelets = (ArrayList<Shapelet>) ois.readObject();
        } catch (IOException | ClassNotFoundException ex) {
            System.out.println(ex);
        }
        
        return shapelets;
    }

    //memUsage is in MB.
    public static void buildGraceBSUB(String fileName, int numInstances, int fold, String queue, int memUsage) {
        try {

            //create the directory and the files.
            File f1 = new File(fileName+"GRACE.bsub");
            f1.createNewFile();

            //write the bsubs
            try (PrintWriter pw = new PrintWriter(f1)) {
                pw.println("#!/bin/csh");
                pw.println("#BSUB -q " + queue);
                pw.println("#BSUB -J " + fileName+fold + "[1-" + numInstances + "]"); //+1 because we have to start at 1.
                pw.println("#BSUB -cwd \"/gpfs/sys/raj09hxu/GraceTransform/dist\"");
                pw.println("#BSUB -oo output/" + fileName+fold + "_%I.out");
                pw.println("#BSUB -R \"rusage[mem=" + memUsage + "]\"");
                pw.println("#BSUB -M " + (memUsage)); //give ourselves a 20% wiggle room.
                pw.println("./etc/profile");
                //pw.println("module add java/jdk/1.7.0_13");
                pw.println("module add java/jdk1.8.0_51");
                pw.println("java -jar -Xmx" + memUsage + "m TimeSeriesClassification.jar " + fileName + " 1 " + (fold+1) + " $LSB_JOBINDEX" );
            }
        } catch (IOException ex) {
            System.out.println("Failed to create file " + ex);
        }
    }

    public static void main(String[] args) {
        
    }
    
    public static void test()
    {
        final String ucrLocation = "../../time-series-datasets/TSC Problems";
        final String transformLocation = "../../";

        String fileExtension = File.separator + DataSets.tscProblemsSmall[0] + File.separator + DataSets.tscProblemsSmall[0];

        Instances train = utilities.ClassifierTools.loadData(ucrLocation + fileExtension + "_TRAIN");
        Instances test = utilities.ClassifierTools.loadData(ucrLocation + fileExtension + "_TEST");

        //first run: build the BSUB.
        //GraceFullShapeletTransform.buildGraceBSUB("../../"+DataSets.tscProblemsSmall[0], train.numInstances(), "raj09hxu", "SamplingExperiments/dist", "samplingExperiments", "long", 1000);
        

        GraceShapeletTransform st = new GraceShapeletTransform();
        st.setNumberOfShapelets(train.numInstances()*10);
        st.setLogOutputFile(DataSets.tscProblemsSmall[0] + ".csv");

        //set the params for your transform. length, shapelets etc.
        
        //second run: using the BSUB. for the cluster
        //st.setSeries(Integer.parseInt(args[0])-1);
        //st.process(train);
        

        //third run: for your own machine. this will build the datasets.
        String classifierDir = File.separator + st.getClass().getSimpleName() + fileExtension;
        String savePath = transformLocation + classifierDir;

//        LocalInfo.saveDataset(st.processFromSubFile(train), savePath + "_TRAIN");
//        LocalInfo.saveDataset(st.process(test), savePath + "_TEST");
        /**/
    }

}
