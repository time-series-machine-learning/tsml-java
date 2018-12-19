/*
 A binary filter that uses information gain quality measure to determine the split point/
     * copyright: Anthony Bagnall
 */
package timeseriesweka.filters;

//import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
import utilities.class_distributions.TreeSetClassDistribution;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author Jon Hills j.hills@uea.ac.uk
 */
public class BinaryTransform extends SimpleBatchFilter{
    private boolean findNewSplits=true;
    private double[] splits;
    public void findNewSplits(){findNewSplits=true;}
    
    
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception{
//Check all are numerical
                 //Check all attributes are real valued, otherwise throw exception
        for(int i=0;i<inputFormat.numAttributes();i++)
                if(inputFormat.classIndex()!=i)
                        if(!inputFormat.attribute(i).isNumeric())
                                throw new Exception("Non numeric attribute not allowed in BinaryTransform");       
        int length=inputFormat.numAttributes();
        if(inputFormat.classIndex()>=0)
            length--;
        
                //Set up instances size and format. 
        FastVector atts=new FastVector();
        FastVector attributeValues=new FastVector();
        attributeValues.addElement("0");
        attributeValues.addElement("1");

        String name;
        for(int i=0;i<length;i++){
                name = "Binary_"+i;
                atts.addElement(new Attribute(name,attributeValues));
        }
        if(inputFormat.classIndex()>=0){	//Classification set, set class 
                //Get the class values as a fast vector			
                Attribute target =inputFormat.attribute(inputFormat.classIndex());

                FastVector vals=new FastVector(target.numValues());
                for(int i=0;i<target.numValues();i++)
                        vals.addElement(target.value(i));
                atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
        }	
        Instances result = new Instances("Binary"+inputFormat.relationName(),atts,inputFormat.numInstances());
        if(inputFormat.classIndex()>=0){
                result.setClassIndex(result.numAttributes()-1);
        }
        return result;       
        
        
    }
    @Override
    public Instances process(Instances data) throws Exception{
         Instances output = determineOutputFormat(data);
         if(findNewSplits){
            splits=new  double[data.numAttributes()];
            double[] classes=new  double[data.numInstances()];
            for(int i=0;i<classes.length;i++)
                classes[i]=data.instance(i).classValue();
            for (int j=0; j< data.numAttributes(); j++) { // for each data
                if(j!=data.classIndex()){

    //Get values of attribute j
                    double[] vals=new double[data.numInstances()];
                    for(int i=0;i<data.numInstances();i++)
                        vals[i]=data.instance(i).value(j);
    //find the IG split point                
                    splits[j] =findSplitValue(data,vals,classes);
                }
            }
            findNewSplits=false;
         }
//Extract out the terms and set the attributes
        for(int i=0;i<data.numInstances();i++){
            Instance newInst=new DenseInstance(data.numAttributes());
            for(int j=0;j<data.numAttributes();j++){
                if(j!=data.classIndex()){
                    if(data.instance(i).value(j)<splits[j])
                        newInst.setValue(j,0);
                    else
                        newInst.setValue(j,1);
                }
                else
                    newInst.setValue(j,data.instance(i).classValue());
            }
            output.add(newInst);
        }
        return output;
    }
    public double findSplitValue(Instances data, double[] vals, double[] classes){
//        return 1;
//Put into an order list
        ArrayList<OrderLineObj> list=new ArrayList<OrderLineObj>();
        for(int i=0;i<vals.length;i++)
            list.add(new OrderLineObj(vals[i],classes[i]));
        //Sort the vals
        TreeSetClassDistribution tree = new TreeSetClassDistribution(data);
        Collections.sort(list);
        return infoGainThreshold(list,tree);
    }
   private static double entropy(TreeSetClassDistribution classDistributions){
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

    public static double infoGainThreshold(ArrayList<OrderLineObj> orderline, TreeSetClassDistribution classDistribution){
// for each split point, starting between 0 and 1, ending between end-1 and end
// addition: track the last threshold that was used, don't bother if it's the same as the last one
        double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
        double thisDist = -1;

        double bsfGain = -1;
        double threshold = -1;

        // check that there is actually a split point
        // for example, if all

        for(int i = 1; i < orderline.size(); i++){
            thisDist = orderline.get(i).getDistance();
            if(i==1 || thisDist != lastDist){ // check that threshold has moved(no point in sampling identical thresholds)- special case - if 0 and 1 are the same dist

                // count class instances below and above threshold
                TreeSetClassDistribution lessClasses = new TreeSetClassDistribution();
                TreeSetClassDistribution greaterClasses = new TreeSetClassDistribution();

                for(double j : classDistribution.keySet()){
                    lessClasses.put(j, 0);
                    greaterClasses.put(j, 0);
                }

                int sumOfLessClasses = 0;
                int sumOfGreaterClasses = 0;

                //visit those below threshold
                for(int j = 0; j < i; j++){
                    double thisClassVal = orderline.get(j).getClassVal();
                    int storedTotal = lessClasses.get(thisClassVal);
                    storedTotal++;
                    lessClasses.put(thisClassVal, storedTotal);
                    sumOfLessClasses++;
                }

                //visit those above threshold
                for(int j = i; j < orderline.size(); j++){
                    double thisClassVal = orderline.get(j).getClassVal();
                    int storedTotal = greaterClasses.get(thisClassVal);
                    storedTotal++;
                    greaterClasses.put(thisClassVal, storedTotal);
                    sumOfGreaterClasses++;
                }

                int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                double parentEntropy = entropy(classDistribution);

                // calculate the info gain below the threshold
                double lessFrac =(double) sumOfLessClasses / sumOfAllClasses;
                double entropyLess = entropy(lessClasses);
                // calculate the info gain above the threshold
                double greaterFrac =(double) sumOfGreaterClasses / sumOfAllClasses;
                double entropyGreater = entropy(greaterClasses);

                double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
//                    System.out.println(parentEntropy+" - "+lessFrac+" * "+entropyLess+" - "+greaterFrac+" * "+entropyGreater);
//                    System.out.println("gain calc:"+gain);
                if(gain > bsfGain){
                    bsfGain = gain;
                    threshold =(thisDist - lastDist) / 2 + lastDist;
                }
            }
            lastDist = thisDist;
        }
        return threshold;
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
       
}
