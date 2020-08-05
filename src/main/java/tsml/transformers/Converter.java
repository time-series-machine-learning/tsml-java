package tsml.transformers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Converter implements Transformer {

    @Override
    public Instance transform(Instance inst) {
        return inst;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        return inst;
    }

    public static TimeSeriesInstances fromArff(Instances data){
        List<List<List<Double>>> raw_data = new ArrayList<>(data.numInstances());
        List<Double> label_indexes =  new ArrayList<>(data.numInstances());

        //we multivariate
        if(data.get(0).attribute(0).isRelationValued()){
            for(int i=0; i<data.numInstances(); i++){            
                Instances timeseries = data.get(i).relationalValue(data.get(i).attribute(0));
                //number of channels is numInstances
                raw_data.add(new ArrayList<>(timeseries.numInstances()));
                for(int j=0; j<timeseries.numInstances(); j++){
                    raw_data.get(i).add(new ArrayList<>(timeseries.numAttributes()));
                    for(int k=0; k< timeseries.get(j).numAttributes(); k++){
                        raw_data.get(i).get(j).add(timeseries.get(j).value(k));
                    }
                }

                label_indexes.add(data.get(i).value(1));
            }
        }
        else{
            for(int i=0; i<data.numInstances(); i++){
                //add dimension 0
                raw_data.add(new ArrayList<>(1));
                raw_data.get(i).add(new ArrayList<>(data.get(i).numAttributes()-1)); //remove class attribute.
                for(int j=0; j< data.get(i).numAttributes(); j++){
                    //skip class index.
                    if(data.classIndex() == j)
                        label_indexes.add(data.get(i).value(j));
                    else
                        raw_data.get(i).get(0).add(data.get(i).value(j));
                }
            }
        }

        String[] labels = new String[data.classAttribute().numValues()];
        for(int i=0; i< labels.length; i++)
            labels[i] = data.classAttribute().value(i);

        TimeSeriesInstances output = new TimeSeriesInstances(raw_data, label_indexes);
        output.setClassLabels(labels);

        return output;
    }


    //TODO: implement singular conversion
    public static TimeSeriesInstance fromArff(Instance data){

        //check if the first attribute is relational.

        //if its multivariate, get class index / value from index 1,
        //loop through all the series in the Instances of the relational attribute.

        //if it its not then convert it to double array, strip out class index and good to go.
        return null;
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

    public static Instance toArff(TimeSeriesInstance data){
        int numChannels = data.getNumChannels();
        if(numChannels == 1)
            return new DenseInstance(1.0, ArrayUtils.add(data.toValueArray()[0], data.getLabelIndex()));


        int numAttributes = data.getMaxLength();
        ArrayList<Attribute> relational_atts = createAttributes(numAttributes);
        Instances relationalHeader =  new Instances("", relational_atts, numChannels);
        relationalHeader.setRelationName("relationalAtt");

        Instance output = new DenseInstance(2);
                                
        //set relation for the dataset/
        Instances relational = new Instances(relationalHeader, numChannels);

        double[][] values = data.toValueArray();
        int classIndex = data.getLabelIndex();

        //each dense instance is row/ which is actually a channel.
        for(int j=0; j< numChannels; j++){
            double[] vals = new double[numAttributes];
            System.arraycopy(values[j], 0, vals, 0, values[j].length);
            for(int k=values[j].length; k<numAttributes; k++)
                vals[k] =  Double.NaN; //all missing values are NaN.
            relational.add(new DenseInstance(1.0, vals));
        }       
                        
        int index = output.attribute(0).addRelation(relational);
        
        //set the relational attribute.
        output.setValue(0, index);           
        
        //set class value.
        output.setValue(1, (double)classIndex);

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

        Transformer conv = new Converter();

        insts_ts = conv.transform(insts_ts);

        insts_arff = conv.transform(insts_arff);
    }
    
}