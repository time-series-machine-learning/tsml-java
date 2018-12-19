package timeseriesweka.filters;

import java.util.ArrayList;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 * Filter to reduce dimensionality of a time series into Piecewise Aggregate Approximation (PAA) form. 
 * Default number of intervals = 8
 *
 * @author James
 */
public class PAA extends SimpleBatchFilter {

    private int numIntervals = 8;
    
    private static final long serialVersionUID = 1L;
    
    public int getNumIntervals() {
        return numIntervals;
    }
    
    public void setNumIntervals(int intervals) {
        numIntervals = intervals;
    }


    @Override
    protected Instances determineOutputFormat(Instances inputFormat)
            throws Exception {
        
        //Check all attributes are real valued, otherwise throw exception
        for (int i = 0; i < inputFormat.numAttributes(); i++) {
            if (inputFormat.classIndex() != i) {
                if (!inputFormat.attribute(i).isNumeric()) {
                    throw new Exception("Non numeric attribute not allowed for PAA");
                }
            }
        }
        
        //Set up instances size and format. 
        FastVector attributes = new FastVector();
        
        for (int i = 0; i < numIntervals; i++)
            attributes.addElement(new Attribute("PAAInterval_" + i));

        if (inputFormat.classIndex() >= 0) {	//Classification set, set class 
            //Get the class values as a fast vector			
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            attributes.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        
        Instances result = new Instances("PAA" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Instances process(Instances input) 
            throws Exception {
        Instances output = determineOutputFormat(input);

        for (int i = 0; i < input.numInstances(); i++) {
            
            double[] data = input.instance(i).toDoubleArray();
            
            //remove class attribute if needed
            double[] temp;
            int c = input.classIndex();
            if(c >= 0) {
                temp=new double[data.length-1];
                System.arraycopy(data,0,temp,0,c); //assumes class attribute is in last index
                data=temp;
            }
            
            double[] intervals = convertInstance(data);
            
            //Now in PAA form, extract out the terms and set the attributes of new instance
            Instance newInstance;
            if (input.classIndex() >= 0)
                newInstance = new DenseInstance(numIntervals + 1);
            else
                newInstance = new DenseInstance(numIntervals);

            for (int j = 0; j < numIntervals; j++)
                newInstance.setValue(j, intervals[j]);
                
            if (input.classIndex() >= 0)
                newInstance.setValue(output.classIndex(), input.instance(i).classValue());

            output.add(newInstance);
        }
        
        return output;
    }
    
    private double[] convertInstance(double[] data) 
            throws Exception {
        
        if (numIntervals > data.length) 
            throw new Exception(
                    "Error converting to PAA, number of intervals (" + numIntervals + ") greater"
                    + " than series length (" + data.length + ")");
        
        double[] intervals = new double[numIntervals];

        //counters to keep track of progress towards completion of a frame
        //potential for data.length % intervals != 0, therefore non-integer
        //interval length, so weight the boundary data points to effect both 
        //intervals it touches
        int currentFrame = 0;
        double realFrameLength = (double)data.length / numIntervals;
        double frameSum = 0.0, currentFrameSize = 0.0, remaining = 0.0;
        
        //PAA conversion
        for (int i = 0; i < data.length; ++i) {
            remaining = realFrameLength - currentFrameSize;

            if (remaining > 1.0) {
                //just use whole data point 
                frameSum += data[i];
                currentFrameSize += 1;
            } else {
                //use some portion of data point as needed
                frameSum += remaining * data[i]; 
                currentFrameSize += remaining;
            }   
            
            if (currentFrameSize == realFrameLength) { //if frame complete
                intervals[currentFrame++] = frameSum / realFrameLength; //store mean

                //before going onto next datapoint, 'use up' any of the current one on the new interval
                //that might not have been used for interval just completed
                frameSum = (1-remaining) * data[i];
                currentFrameSize = (1-remaining);
            }
        }
        
        // i.e. if the last interval didn't write because of double imprecision
        if (currentFrame == numIntervals-1) { //if frame complete
            intervals[currentFrame++] = frameSum / realFrameLength;
        }
        
        return intervals;
    }
    
    public static double[] convertInstance(double[] data, int numIntervals) throws Exception {
        PAA paa = new PAA();
        paa.setNumIntervals(numIntervals);
        
        return paa.convertInstance(data);
    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) {
//        System.out.println("PAAtest\n\n");
//        
//        try {
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
//            PAA paa = new PAA();
//            paa.setNumIntervals(2);
//            Instances result = paa.process(test);
//            
//            System.out.println(test);
//            System.out.println("\n\n\nResults:\n\n");
//            System.out.println(result);
//        }
//        catch (Exception e) {
//            System.out.println(e);
//        }

        
        // Jason's Test
        try{
            double[] wavey = {0.841470985,0.948984619,0.997494987,0.983985947,0.909297427,0.778073197,0.598472144,0.381660992,0.141120008,-0.108195135,-0.350783228,-0.571561319,-0.756802495,-0.894989358,-0.977530118,-0.999292789,-0.958924275,-0.858934493,-0.705540326,-0.508279077,-0.279415498};
            
            PAA paa = new PAA();
            paa.setNumIntervals(10);
            
            // convert into Instances format            
            ArrayList<Attribute> atts = new ArrayList<>();
            DenseInstance ins = new DenseInstance(wavey.length+1);
            for(int i = 0; i < wavey.length; i++){
                ins.setValue(i, wavey[i]);
                atts.add(new Attribute("att"+i));
            }
            atts.add(new Attribute("classVal"));
            ins.setValue(wavey.length, 1);

            Instances instances = new Instances("blah", atts, 1);
            instances.setClassIndex(instances.numAttributes()-1);
            instances.add(ins);
                        
            Instances out = paa.process(instances);

            for(int i = 0; i < out.numAttributes()-1;i++){
                System.out.print(out.instance(0).value(i)+",");
            }
            System.out.println();
        }catch(Exception e){
            e.printStackTrace();
        }

    }

}
