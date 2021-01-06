/** Class NormalizeAttribute.java
 * 
 * @author AJB
 * @version 1
 * @since 14/4/09
 * 
 * Class normalizes attributes, basic version. Assumes no missing values. 
 * 
 * Normalise onto [0,1] if norm==NormType.INTERVAL, 
 * Normalise onto Normal(0,1) if norm==NormType.STD_NORMAL, 
 * 
 * 
 */

package tsml.transformers;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.InstanceTools;
import utilities.NumUtils;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class RowNormalizer implements Transformer {
	public enum NormType {
		INTERVAL, STD, STD_NORMAL
	};

	public static boolean throwErrorOnZeroVariance = false;

	NormType norm = NormType.STD_NORMAL;

	public RowNormalizer(){
		this(NormType.STD_NORMAL);
	}

	public RowNormalizer(NormType type){
		norm = type;
	}

	@Override
	public Instances transform(Instances data) {
		switch (norm) {
			case INTERVAL: // Map onto [0,1]
				return intervalNorm(data);
			case STD: // Subtract the mean of the series
				return standard(data);
			default: // Transform to zero mean, unit variance
				return standardNorm(data);
		}
	}

	@Override
	public Instance transform(Instance inst) {
		switch (norm) {
			case INTERVAL: // Map onto [0,1]
				return intervalNorm(inst);
			case STD: // Subtract the mean of the series
				return standard(inst);
			default: // Transform to zero mean, unit variance
				return standardNorm(inst);
		}
	}

	/* Wont normalise the class value */
	public Instances intervalNorm(Instances data) {
		Instances out = determineOutputFormat(data);
		for(Instance inst : data){
			out.add(intervalNorm(inst));
		}

		return out;
	}

	public Instance intervalNorm(Instance inst) {
		return intervalInCopy(inst, InstanceTools.max(inst), InstanceTools.min(inst));
	}



	public Instances standard(Instances data) {
		Instances out = determineOutputFormat(data);
		for(Instance inst : data){
			out.add(standard(inst));
		}

		return out;
	}

	public Instance standard(Instance data){
		double size = data.numAttributes();
		int classIndex = data.classIndex();
		if (classIndex > 0)
			size--;
		double mean = InstanceTools.sum(data) / size;

		return standardInCopy(data, mean);
	}

	public Instances standardNorm(Instances data) {
		Instances out = determineOutputFormat(data);
		for(Instance inst : data){
			out.add(standardNorm(inst));
		}

		return out;
	}

	private Instance standardNorm(Instance inst) {
		double mean,sum,sumSq,var = 0;
		double size = inst.numAttributes();
		if (inst.classIndex() >= 0)
			size--;
		sum = InstanceTools.sum(inst);
		sumSq = InstanceTools.sumSq(inst);

		var = (sumSq - sum * sum / size) / size;
		mean = sum / size;

		//if the instance has zero variance just return a list of 0's, else return the normed instance.
		return NumUtils.isNearlyEqual(var,0) ? constantInCopy(inst, 0) : normaliseInCopy(inst, mean, Math.sqrt(var));
	}
	@Override
	public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
		FastVector<Attribute> atts=new FastVector<>();

		for(int i=0;i<inputFormat.numAttributes()-1;i++)
		{                         
			String name = norm.toString()+i;
			atts.addElement(new Attribute(name));
		}
		//Get the class values as a fast vector			
		Attribute target =inputFormat.attribute(inputFormat.classIndex());

		FastVector<String> vals=new FastVector<>(target.numValues());
		for(int i=0;i<target.numValues();i++)
				vals.addElement(target.value(i));
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
		Instances result = new Instances("NormaliseCase "+inputFormat.relationName(),atts,inputFormat.numInstances());
		if(inputFormat.classIndex()>=0){
				result.setClassIndex(result.numAttributes()-1);
		}

		return result;
	}



	/*I THINK THESE COULD BE REFACTORED INTO INSTANCE TOOLS*/
    public static Instance normaliseInCopy(Instance orig, double mean, double std){
        Instance copy = new DenseInstance(orig.numAttributes());
        for (int j = 0; j < orig.numAttributes(); j++) 
            if (j != orig.classIndex() && !orig.attribute(j).isNominal()) 
                copy.setValue(j, (orig.value(j) - mean) / (std));

        if(orig.classIndex() >= 0)
            copy.setValue(orig.classIndex(), orig.classValue());

        return copy;
    }

    public static Instance standardInCopy(Instance orig, double mean){
        Instance copy = new DenseInstance(orig.numAttributes());
        for (int j = 0; j < orig.numAttributes(); j++) 
            if (j != orig.classIndex() && !orig.attribute(j).isNominal()) 
                copy.setValue(j, (orig.value(j) - mean));

        if(orig.classIndex() >= 0)
            copy.setValue(orig.classIndex(), orig.classValue());

        return copy;
    }

    public static Instance intervalInCopy(Instance orig, double min, double max){
        Instance copy = new DenseInstance(orig.numAttributes());
        for (int j = 0; j < orig.numAttributes(); j++) 
            if (j != orig.classIndex() && !orig.attribute(j).isNominal()) 
                copy.setValue(j, (orig.value(j) - min) / (max - min));

        if(orig.classIndex() >= 0)
            copy.setValue(orig.classIndex(), orig.classValue());

        return copy;
    }

    public static Instance constantInCopy(Instance orig, double constant){
        Instance copy = new DenseInstance(orig.numAttributes());
        for (int j = 0; j < orig.numAttributes(); j++) 
            if (j != orig.classIndex() && !orig.attribute(j).isNominal()) 
                copy.setValue(j, constant);

        if(orig.classIndex() >= 0)
            copy.setValue(orig.classIndex(), orig.classValue());

        return copy;
	}
	/* END BLOCK */


	/* TimeSeries Utilities */

	@Override
	public TimeSeriesInstances transform(TimeSeriesInstances data) {
		switch (norm) {
			case INTERVAL: // Map onto [0,1]
				return intervalNorm(data);
			case STD: // Subtract the mean of the series
				return standard(data);
			default: // Transform to zero mean, unit variance
				return standardNorm(data);
		}
	}

	public static TimeSeriesInstances standardNorm(TimeSeriesInstances data) {
		TimeSeriesInstances out = new TimeSeriesInstances(data.getClassLabels());
		for(TimeSeriesInstance inst : data){
			out.add(standardNorm(inst));
		}
		
		return out;
	}

	public static TimeSeriesInstances standard(TimeSeriesInstances data) {
		TimeSeriesInstances out = new TimeSeriesInstances(data.getClassLabels());
		for(TimeSeriesInstance inst : data){
			out.add(standard(inst));
		}
		
		return out;
	}

	public static TimeSeriesInstances intervalNorm(TimeSeriesInstances data) {
		TimeSeriesInstances out = new TimeSeriesInstances(data.getClassLabels());
		for(TimeSeriesInstance inst : data){
			out.add(intervalNorm(inst));
		}
		
		return out;
	}

	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
		switch (norm) {
			case INTERVAL: // Map onto [0,1]
				return intervalNorm(inst);
			case STD: // Subtract the mean of the series
				return standard(inst);
			default: // Transform to zero mean, unit variance
				return standardNorm(inst);
		}
	}

	public static TimeSeriesInstance standardNorm(TimeSeriesInstance inst) {
		List<TimeSeries> out = new ArrayList<>(inst.getNumDimensions());
		for(TimeSeries ts : inst){
			out.add(standardNorm(ts));
		}
		
		return new TimeSeriesInstance(inst.getLabelIndex(), inst.getClassLabels(), out);
	}

	public static TimeSeriesInstance standard(TimeSeriesInstance inst) {
		List<TimeSeries> out = new ArrayList<>(inst.getNumDimensions());
		for(TimeSeries ts : inst){
			out.add(standard(ts));
		}
		
		return new TimeSeriesInstance(inst.getLabelIndex(), inst.getClassLabels(), out);
	}

	public static TimeSeriesInstance intervalNorm(TimeSeriesInstance inst) {
		List<TimeSeries> out = new ArrayList<>(inst.getNumDimensions());
		for(TimeSeries ts : inst){
			out.add(intervalNorm(ts));
		}
		
		return new TimeSeriesInstance(inst.getLabelIndex(), inst.getClassLabels(), out);
	}

	public static TimeSeries standardNorm(TimeSeries ts) {
		double[] out = ts.toValueArray(); //this is a copy.

		double mean = TimeSeriesSummaryStatistics.mean(out);
		double var = TimeSeriesSummaryStatistics.variance(ts, mean);
		boolean constant = NumUtils.isNearlyEqual(var,0);

		//if we have zero variance, then just return array of 0's
		if(constant){
			for(int i =0; i<out.length; i++){
				out[i] = 0;
			}
		}
		else{
			double std = Math.sqrt(var);

			for(int i =0; i<out.length; i++){
				out[i] =  (out[i] - mean) / std;
			}
		}
		return new TimeSeries(out);
	}


	public static TimeSeries standard(TimeSeries ts){
		double[] out = ts.toValueArray(); //this is a copy.

		double mean = TimeSeriesSummaryStatistics.mean(out);

		for(int i =0; i<out.length; i++){
			out[i] =  (out[i] - mean);
		}

		return new TimeSeries(out);
	}

	public static TimeSeries intervalNorm(TimeSeries ts){
		double[] out = ts.toValueArray(); //this is a copy.

		double max = TimeSeriesSummaryStatistics.max(out);
		double min =  TimeSeriesSummaryStatistics.min(out);

		for(int i =0; i<out.length; i++){
			out[i] =  (out[i] - min) / (max - min);
		}

		return new TimeSeries(out);
	}


	
	static String[] fileNames = { // Number of train,test cases,length,classes
			"Beef", // 30,30,470,5
			"Coffee", // 28,28,286,2
			"OliveOil", "Earthquakes", "Ford_A", "Ford_B" };
	static String path = "C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

	public static void main(String[] args) {
		String local_path = "D:\\Work\\Data\\Univariate_ts\\"; //Aarons local path for testing.
        String dataset_name = "ChinaTown";
        Instances train = DatasetLoading.loadData(local_path + dataset_name + File.separator + dataset_name+"_TRAIN.ts");
        Instances test  = DatasetLoading.loadData(local_path + dataset_name + File.separator + dataset_name+"_TEST.ts");
        RowNormalizer hTransform= new RowNormalizer();
        Instances out_train = hTransform.transform(train);
        Instances out_test = hTransform.transform(test);
        System.out.println(out_train.toString());
        System.out.println(out_test.toString());
	}



	
	
}
