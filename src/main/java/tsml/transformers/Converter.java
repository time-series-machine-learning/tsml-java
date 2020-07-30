package tsml.transformers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Converter implements TrainableTransformer {

    boolean isFit = false;


    Instances input = null;
    TimeSeriesInstances inputTS = null;


    @Override
    public Instance transform(Instance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    @Override
    public void fit(Instances data) {
        input = data;
        isFit = true;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        inputTY = data;
        isFit = true;
    }


    public static TimeSeriesInstances fromArff(Instances data){

    }

    public static Instances toArff(TimeSeriesInstances  data){
        double[][][] values = data.toValueArray();
        int[] classIndexes = data.getClassIndexes();
        String[] classLabels = data.getClassLabels();

        int numAttributes = data.getMaxLength();
        int numChannels = data.getMaxNumChannels();

        if(data.isMultivariate()){
            //create relational attributes.
            ArrayList<Attribute> relational_atts = createAttributes(numAttributes);
            Instances relationalHeader =  new Instances("", relational_atts, numChannels);
            relationalHeader.setRelationName("relationalAtt");

            //create the relational and class value attributes.
            ArrayList<Attribute> attributes = new ArrayList<>();
            Attribute relational_att = new Attribute("relationalAtt", relationalHeader, 0);        
            attributes.add(relational_att);
            attributes.add(new Attribute("ClassLabel", Arrays.stream(classLabels).collect(Collectors.toList())));
            
            //create output data set.
            Instances output = new Instances("Converted", attributes, data.numInstances());
                        
            for(int i=0; i < data.numInstances(); i++){
                //create each row.
                //only two attribtues, relational and class.
                output.add(new DenseInstance(2));
                                
                //set relation for the dataset/
                Instances relational = new Instances(relationalHeader, data.get(i).getNumChannels());
        
                //each dense instance is row/ which is actually a channel.
                for(int j=0; j< data.get(i).getNumChannels(); j++){
                    double[] vals = new double[numAttributes];
                    System.arraycopy(values[i][j], 0, vals, 0, values[i][j].length);
                    for(int k=values[i][j].length; k<numAttributes; k++)
                        vals[k] =  Double.NaN; //all missing values are NaN.
                    relational.add(new DenseInstance(1.0, vals));
                }       
                                
                int index = output.instance(i).attribute(0).addRelation(relational);
                
                //set the relational attribute.
                output.instance(i).setValue(0, index);           
                
                //set class value.
                output.instance(i).setValue(1, (double)classIndexes[i]);
            }
            
            output.setClassIndex(output.numAttributes()-1);
            //System.out.println(relational);
            return output; 
        }

        //if its not multivariate its univariate.    
        ArrayList<Attribute> attributes = createAttributes(numAttributes);
        //add the class label at the end.
        attributes.add(new Attribute("ClassLabel", Arrays.stream(classLabels).collect(Collectors.toList())));

        //TODO: put the dataset name in the TSInstances
        Instances output = new Instances("Converted", attributes, data.numInstances());
        output.setClassIndex(output.numAttributes() - 1);

        //create the Instance.
        for (int i = 0; i < data.numInstances(); i++) {
            //we know it's univariate so it has only one dimension.
            double[] vals = new double[numAttributes+1];
            System.arraycopy(values[i][0], 0, vals, 0, values[i][0].length);
            for(int j=values[i][0].length; j<numAttributes; j++)
                vals[j] =  Double.NaN; //all missing values are NaN.
            vals[vals.length-1] = (double)classIndexes[i]; //put class val at the end.
            output.add(new DenseInstance(1.0, vals));
        }

        return output;
            

    }

    private static ArrayList<Attribute> createAttributes(int numAttributes) {
        ArrayList<Attribute> relational_atts = new ArrayList<>();
        for (int i = 0; i < numAttributes; i++) {
            relational_atts.add(new Attribute("TimeSeriesData_" + i));
        }
        return relational_atts;
    }

    public static void main(String[] args) {
        
        TimeSeriesInstances insts_ts = null;
        Instances insts_arff = null;

        TrainableTransformer conv = new Converter();

        conv.fit(insts_arff);
        insts_ts = conv.transform(insts_ts);
        
        conv.fit(insts_ts);
        insts_arff = conv.transform(insts_arff);
    }
    
}