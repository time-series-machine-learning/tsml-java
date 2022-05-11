package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import java.io.FileReader;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    private boolean useGain = false;
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        int count = data.numClasses();
        int value = att.numValues();

        if(att.isNumeric()){
            Instances[] splitData = splitDataOnNumeric(data,att);
            int[][] contingencyTable = new int[2][count];
            for (int i=0; i<2;i++){
                for(Instance instance:splitData[i]){
                    value = (int)instance.classValue();
                    contingencyTable[i][value]++;
                }
            }
            if (useGain){
                return AttributeMeasures.measureInformationGainRatio(contingencyTable);
            } else {
                return AttributeMeasures.measureInformationGain(contingencyTable);
            }
        }else{

            int[][] contingencyTable = new int[value][count];
            System.out.println(value);
            System.out.println(count);

            for (Instance instance : data){
                int attributeValue = (int) instance.value(att);
                int classValue = (int) instance.classValue();
                contingencyTable[attributeValue][classValue]++;
            }
            if (useGain){
                return AttributeMeasures.measureInformationGainRatio(contingencyTable);
            } else {
                return AttributeMeasures.measureInformationGain(contingencyTable);
            }
        }
    }

    public void setUseGain(boolean useGain){
        this.useGain = useGain;
    }
    public static void main(String[] args){
        try{
            FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/Whisky.arff");
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);

            IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();
            for (int i=0; i<data.numAttributes()-1;i++){
                double infGain = igAttributeSplitMeasure.computeAttributeQuality(data, data.attribute(i));
                System.out.println("measure Information Gain for attribute "+data.attribute(i).name()+" splitting diagnosis "+infGain);
            }

            igAttributeSplitMeasure.useGain = true;

            for (int i=0; i<data.numAttributes()-1;i++){
                double infGain = igAttributeSplitMeasure.computeAttributeQuality(data, data.attribute(i));
                System.out.println("measure Information Gain Ratio for attribute "+data.attribute(i).name()+" splitting diagnosis "+infGain);
            }
    }
        catch (Exception e) {
            System.out.println("Error: "+e);
        }
    }
}
