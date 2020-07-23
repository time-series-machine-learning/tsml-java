package tsml.transformers;

import java.util.ArrayList;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;
import utilities.NumUtils;

public class ConstantAttributeRemover implements TrainableTransformer {

    ArrayList<Integer> attsToRemove;
    boolean isFit;

    int minChecksToDiscard = 3;

    private boolean IsAttributeConstant(final Instances data, final int attToCheck) {

        final double firstVal = data.firstInstance().value(attToCheck);
        for (int i = 1; i < data.numInstances(); i++) {
            if (!NumUtils.isNearlyEqual(firstVal, data.get(i).value(attToCheck)))
                return false;
        }
        return true;
    }

    private ArrayList<Integer> FindConstantAtts(final Instances data) {

        ArrayList<Integer> out = new ArrayList<>();
        // loop through all attributes from the end.
        for (int i = data.numAttributes() - 1; i >= 0; --i) {
            if (IsAttributeConstant(data, i)) {
                out.add(i);
            }
        }
        return out;
    }



    private boolean isAttributeConstant(final TimeSeriesInstances data, final int attToCheck){
        //in the first series, in the first dimension, get the att to check.
        final double firstVal = data.get(0).get(0).get(attToCheck);
        int count =0;
        for(TimeSeriesInstance inst : data){
            for(TimeSeries ts : inst){
                //if a single value, across any dimension is different then not constant.
                //need to deal with odd length series. 
                if(ts.hasValidValueAt(attToCheck))
                    continue;

                if (!NumUtils.isNearlyEqual(firstVal, ts.get(attToCheck))) 
                    return false;
                else
                    count++;
            }
        }
        //if we've been checking atts, need to make sure in unequal arrays we have checked against atleast one other.
        return count >= minChecksToDiscard;
    }

    
    private ArrayList<Integer> FindConstantAtts(final TimeSeriesInstances data) {

        ArrayList<Integer> out = new ArrayList<>();
        // loop through all attributes from the end.
        
        for (int i = data.getMaxLength() - 1; i >= 0; --i) {
            if (isAttributeConstant(data, i)) {
                out.add(i);
            }
        }
        return out;
    }

    @Override
    public void fit(final Instances data) {
        attsToRemove = FindConstantAtts(data);
        isFit = true;
    }


    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        inst.to


        return null;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        attsToRemove = FindConstantAtts(data);
        isFit = true;
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    @Override
    public Instances transform(final Instances data) {
        // could clone the instances.
        for (final int att : attsToRemove)
            data.deleteAttributeAt(att);

        return data;
    }

    @Override
    public Instance transform(final Instance inst) {
        // could clone the instances.
        for (final int att : attsToRemove)
            inst.deleteAttributeAt(att);

        return inst;
    }

    public static void main(final String[] args) {

        final double[][] t1 = { { 1, 0, 1, 0.00000000000000004 }, { 2, 0, 2, 0 }, { 3, 0, 3, 0 },
                { 2, 0, 2, 0.000000000000000000001 } };
        final double[][] t2 = { { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 3, 3, 3, 3 }, { 4, 4, 4, 4 } };
        final Instances train = InstanceTools.toWekaInstances(t1);
        final Instances test = InstanceTools.toWekaInstances(t2);

        final ConstantAttributeRemover rr = new ConstantAttributeRemover();
        final Instances out_train = rr.fitTransform(train);
        final Instances out_test = rr.transform(test);

        System.out.println(out_train);
        System.out.println(out_test);

    }


}