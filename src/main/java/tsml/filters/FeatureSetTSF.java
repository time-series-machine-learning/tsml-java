package tsml.filters;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

import java.util.ArrayList;

public class FeatureSetTSF extends SimpleBatchFilter {
    private int numMoments = 3;

    public void setNumMoments(int m) {
        numMoments = 3;
    }

    private static final long serialVersionUID = 1L;

    protected Instances determineOutputFormat(Instances inputFormat)
            throws Exception {
        //Check all attributes are real valued, otherwise throw exception
        for (int i = 0; i < inputFormat.numAttributes(); i++)
            if (inputFormat.classIndex() != i)
                if (!inputFormat.attribute(i).isNumeric())
                    throw new Exception("Non numeric attribute not allowed in SummaryStats");
        //Set up instances size and format.
        ArrayList<Attribute> atts = new ArrayList();
        String source = inputFormat.relationName();
        String name;
        for (int i = 0; i < numMoments; i++) {
            name = source + "Moment_" + (i + 1);
            atts.add(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0) {    //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>(target.numValues());
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Moments" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }


    @Override
    public String globalInfo() {

        return null;
    }

    @Override
    public Instances process(Instances inst) throws Exception {
        Instances output = determineOutputFormat(inst);
        //For each data, first extract the relevan
        int seriesLength = inst.numAttributes();
        if (inst.classIndex() >= 0) {
            seriesLength--;
        }
        for (int i = 0; i < inst.numInstances(); i++) {
            //1. Get series:
            double[] d = inst.instance(i).toDoubleArray();
            //2. Remove target class
            double[] temp;
            int c = inst.classIndex();
            if (c >= 0) {
                temp = new double[d.length - 1];
                System.arraycopy(d, 0, temp, 0, c);
                //                       if(c<temp.length)
                //                           System.arraycopy(d,c+1,temp,c,d.length-(c+1));
                d = temp;
            }
            double[] moments = new double[numMoments + 2];
/**
 *
 *
 * HERE FIND MOMENTS HERE
 *
 *
 **/

        double mean;
        double stDev;
        double slope;

            double sumX=0,sumYY=0;
            double sumY3=0,sumY4=0;
            double sumY=0,sumXY=0,sumXX=0;
            int length=d.length;
            for(int j=0;j<length;j++){
                sumY+=d[j];
                sumYY+=d[j]*d[j];
                sumX+=(j);
                sumXX+=(j)*(j);
                sumXY+=d[j]*(j);
            }
            mean=sumY/length;
            stDev=sumYY-(sumY*sumY)/length;
            slope=(sumXY-(sumX*sumY)/length);
            double denom=sumXX-(sumX*sumX)/length;
            if(denom!=0)
                slope/=denom;
            else
                slope=0;
            stDev/=length;
            if(stDev==0)    //Flat line
                slope=0;
//            else //Why not doing this? Because not needed?
//                stDev=Math.sqrt(stDev);
            if(slope==0)
                stDev=0;

            moments[0] = mean;
            moments[1] = stDev;
            moments[2] = slope;

            //slope

            //Extract out the terms and set the attributes
            Instance newInst = null;
            if (inst.classIndex() >= 0)
                newInst = new DenseInstance(numMoments + 1);
            else
                newInst = new DenseInstance(numMoments);

            for (int j = 0; j < numMoments; j++) {
                newInst.setValue(j, moments[j]);
            }
            if (inst.classIndex() >= 0)
                newInst.setValue(output.classIndex(), inst.instance(i).classValue());
            output.add(newInst);
        }
        return output;
    }
}