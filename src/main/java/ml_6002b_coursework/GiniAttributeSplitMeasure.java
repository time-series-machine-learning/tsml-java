package ml_6002b_coursework;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import java.io.FileReader;

public class GiniAttributeSplitMeasure extends AttributeSplitMeasure {

    /**
     * Checks quality of data for Gini
     * Checks whether numeric or nominal - if numeric perform numeric split
     * Creates contingency table based on data
     * @param data
     * @param att
     * @return Information Gain or Information Gain Ratio based upon setUseGain
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        int count = data.numClasses();
        int value = att.numValues();
        if (att.isNumeric()) {
            Instances[] splitData = splitDataOnNumeric(data, att);
            int[][] contingencyTable = new int[2][count];
            for (int i = 0; i < 2; i++) {
                for (Instance instance : splitData[i]) {
                    value = (int) instance.classValue();
                    contingencyTable[i][value]++;
                }
            }
            return AttributeMeasures.measureGini(contingencyTable);
        }else{
            int[][] contingencyTable = new int[value][count];

            for (Instance instance : data) {
                int attributeValue = (int) instance.value(att);
                int classValue = (int) instance.classValue();
                contingencyTable[attributeValue][classValue]++;
            }
            return AttributeMeasures.measureGini(contingencyTable);
        }

    }

    /**
     * test harness for all different split possibilities
     */
    public static void main(String[] args){
        try {
            FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/Whisky.arff");
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);

            GiniAttributeSplitMeasure giniAttributeSplitMeasure = new GiniAttributeSplitMeasure();
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gini = giniAttributeSplitMeasure.computeAttributeQuality(data, data.attribute(i));
                System.out.println("measure Gini for attribute " + data.attribute(i).name() + " splitting diagnosis " + gini);
            }
        }
        catch (Exception e) {
            System.out.println("Error: "+e);
        }
    }
}

