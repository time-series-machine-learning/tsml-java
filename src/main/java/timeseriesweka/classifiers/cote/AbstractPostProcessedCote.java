package timeseriesweka.classifiers.cote;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public abstract class AbstractPostProcessedCote {

    protected String resultsDir;
    protected String datasetName;
    protected int resampleId = 0;;
    protected ArrayList<String> classifierNames;
    
    protected double[] cvAccs;        // [classifier]
    protected double[] testAccs;      // [classifier]
    protected double[][] testPreds;   // [classifier][instance]
    protected double[][][] testDists; // [classifier][instance][classVal]

    private double[] testActualClassVals;
    
    public static String CLASSIFIER_NAME = "AbstractCOTE";
    
    public void loadResults() throws Exception{
        cvAccs = new double[classifierNames.size()];
        testPreds = new double[classifierNames.size()][];
        testDists = new double[classifierNames.size()][][];
        int testSetSize = -1; // we can work this out when we process the first of the test results
        int numClassVals = -1;
        File trainResult, testResult;
        Scanner scan;
        String[] lineParts;
        int counter;
        for(int c = 0; c < this.classifierNames.size(); c++){
            
            trainResult = new File(resultsDir+classifierNames.get(c)+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv");
            if(!trainResult.exists() || trainResult.length() == 0){
                throw new Exception("Error: training results do not exist ("+trainResult.getAbsolutePath()+")");
            }
            scan = new Scanner(trainResult);
            scan.useDelimiter("\n");
            scan.next();
            scan.next();
            cvAccs[c] = Double.parseDouble(scan.next().trim());
            scan.close();
            // we don't need the cv predictions for anything in the current COTE configs, can address this later if we need to store them though
            
            testResult = new File(resultsDir+classifierNames.get(c)+"/Predictions/"+datasetName+"/testFold"+resampleId+".csv");
            
            if(testSetSize<=0){
                scan = new Scanner(testResult);
                scan.useDelimiter("\n");
                scan.next();    // header
                scan.next();    // param info
                scan.next();    // test acc
                testSetSize = 0;
                while(scan.hasNext()){
                    if(numClassVals<=0){
                        lineParts = scan.next().split(",");
                        numClassVals = lineParts.length-3; // subtract actual, pred, and empty cell for padding
                    }else{
                        scan.next();
                    }
                    testSetSize++;
                }
                scan.close();
            }
            
            if(c==0){
                testPreds = new double[classifierNames.size()][testSetSize];
                testDists = new double[classifierNames.size()][testSetSize][numClassVals];
                testActualClassVals = new double[testSetSize];
                testAccs = new double[classifierNames.size()];
            }
            
            scan = new Scanner(testResult);
            scan.useDelimiter("\n");
            scan.next();
            scan.next();
            testAccs[c] = Double.parseDouble(scan.next().trim());
            counter = 0;
            while(scan.hasNext()){                
                lineParts = scan.next().split(",");
                if(lineParts.length==1){ //Tony's RISE files have rogue lines at the end (sometimes!)
                    continue;
                }
                testPreds[c][counter] = Double.parseDouble(lineParts[1].trim());
                for(int d = 0; d < numClassVals; d++){
                    testDists[c][counter][d] = Double.parseDouble(lineParts[d+3].trim());
                }
                
                if(c==0){
                    testActualClassVals[counter] = Double.parseDouble(lineParts[0].trim());
                }else{
                    if(testActualClassVals[counter]!=Double.parseDouble(lineParts[0].trim())){
                        throw new Exception("Error: class value mismatch. Test file for "+classifierNames.get(c)+ " states that instance "+counter+" has the class value of "+lineParts[0]+", but in "+classifierNames.get(0)+" it was "+testActualClassVals[counter]+".");
                    }
                }
                counter++;
            }
            scan.close();   
        }
    }
    
    
    protected double classifyInstanceFromDistribution(double[] dist){
        double bsfClassVal = -1;
        double bsfClassWeight = -1;
        
        for(int d = 0; d < dist.length; d++){
            if(dist[d] > bsfClassWeight){
                bsfClassWeight = dist[d];
                bsfClassVal = d;
            }
        }
        return bsfClassVal;
    }
    
    public abstract double[] distributionForInstance(int testInstanceId) throws Exception;
    
    public void writeTestSheet() throws Exception{
        writeTestSheet(this.resultsDir);
    }
    public void writeTestSheet(String outputDir) throws Exception{
        
        File outputPath = new File(outputDir+CLASSIFIER_NAME+"/Predictions/"+datasetName+"/");
        if(!outputPath.exists()){
            outputPath.mkdirs();
            if(!outputPath.exists()){
                throw new Exception("Error: invalid results path ("+outputPath+").");
            }
        }
        
        if(cvAccs==null){
            loadResults();
        }
        
        StringBuilder st = new StringBuilder();
        int correct = 0;
        double act, pred;
        double[] dist;
        for(int i = 0; i < testPreds[0].length; i++){
            dist = this.distributionForInstance(i);
            act = this.testActualClassVals[i];
            pred = this.classifyInstanceFromDistribution(dist);
            if(act==pred){
                correct++;
            }
            st.append(act+","+pred+",");
            for(int d = 0; d < dist.length; d++){
                st.append(","+dist[d]);
            }
            st.append("\n");
        }
        FileWriter out = new FileWriter(outputDir+CLASSIFIER_NAME+"/Predictions/"+datasetName+"/testFold"+resampleId+".csv");
        out.append(CLASSIFIER_NAME+","+datasetName+",test\n");
        out.append("constituentCvAccs");
        for(int c = 0; c < classifierNames.size(); c++){
            out.append(","+classifierNames.get(c));
            out.append(","+cvAccs[c]);
        }
        out.append("\n"+((double)correct/testPreds[0].length+"\n"));
        
        out.append(st);
        out.close();
        
    }
    
    
    public double[] getHiveTestPredictions() throws Exception{
        return null;
    }
    
}
