/** Class NormalizeAttribute.java
 * 
 * @author AJB
 * @version 1
 * @since 14/4/09
 * 
 * Class normalizes attributes, basic version. 
 		1. Assumes no missing values. 
 		2. Assumes all attributes real values
 		3. Assumes class index same in all data (vague checks made) but can be none set (classIndex==-1)
 		4. Batch process, by default it calculates the ranges from the instances in trainData, then uses
 		this to process the instances passed. Note that this may produce values outside the 
 		interval range, since the min or max of the test data may be separate. If you want to avoid
 		this, the only way at the moment is to first merge train and test, then pass the merged set.
 		Easy to hack round this if I have to.
 
 * Normalise onto [0,1] if norm==NormType.INTERVAL, 
 * Normalise onto Normal(0,1) if norm==NormType.STD_NORMAL, 
 * 
 * Useage:
 * 	Instances train = //Get Train
 * 	Instances test = //Get Train
 * 
 * NormalizeAttributes na = new NormalizeAttributes(train);
 *
 * na.setNormMethod(NormalizeAttribute.NormType.INTERVAL); //Defaults to interval anyway
 	try{
 //Both	processed with the stats from train.
 		Instances newTrain=na.process(train);
 		Instances newTest=na.process(test);
 		
 */

package tsml.transformers;

import org.apache.commons.lang3.NotImplementedException;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;

public class ColumnNormalizer implements TrainableTransformer {
	enum NormType {
		INTERVAL, STD_NORMAL
	};

	Instances trainData;
	double[] min;
	double[] max;
	double[] mean;
	double[] stdev;
	int classIndex;
	NormType norm = NormType.INTERVAL;
	boolean isFit;

	public ColumnNormalizer() {
	}

	public ColumnNormalizer(Instances data) {
		trainData = data;
		classIndex = data.classIndex();
		// Finds all the stats, doesnt cost much more really
		findStats(data);
	}

	protected void findStats(Instances r) {
		// Find min and max
		// assert(classIndex==r.classIndex());

		max = new double[r.numAttributes()];
		min = new double[r.numAttributes()];
		for (int j = 0; j < r.numAttributes(); j++) {
			max[j] = Double.MIN_VALUE;
			min[j] = Double.MAX_VALUE;
			for (int i = 0; i < r.numInstances(); i++) {
				double x = r.instance(i).value(j);
				if (x > max[j])
					max[j] = x;
				if (x < min[j])
					min[j] = x;
			}
		}

		// Find mean and stdev
		mean = new double[r.numAttributes()];
		stdev = new double[r.numAttributes()];
		double sum, sumSq, x, y;
		for (int j = 0; j < r.numAttributes(); j++) {
			sum = 0;
			sumSq = 0;
			for (int i = 0; i < r.numInstances(); i++) {
				x = r.instance(i).value(j);
				sum += x;
				sumSq += x * x;
			}
			stdev[j] = sumSq / r.numInstances() - sum * sum;
			mean[j] = sum / r.numInstances();
			stdev[j] = Math.sqrt(stdev[j]);
		}
	}

	protected void findStats(TimeSeriesInstances r) {
		max = new double[r.getMaxLength()];
		min = new double[r.getMaxLength()];
		mean = new double[r.getMaxLength()];
		stdev = new double[r.getMaxLength()];

		for (int j = 0; j < r.getMaxLength(); j++) {
			double[] slice = r.getVSliceArray(j);

			max[j] = TimeSeriesSummaryStatistics.max(slice);
			min[j] = TimeSeriesSummaryStatistics.min(slice);
			mean[j] = TimeSeriesSummaryStatistics.mean(slice);
			stdev[j] = Math.sqrt(TimeSeriesSummaryStatistics.variance(slice, mean[j]));
		}
	}

	public double[] getRanges() {
		double[] r = new double[max.length];
		for (int i = 0; i < r.length; i++)
			r[i] = max[i] - min[i];
		return r;
	}

	@Override
	public Instance transform(Instance inst) {
		throw new NotImplementedException("Column wise normalisation doesn't make sense for single instances");
	}

	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
		throw new NotImplementedException("Column wise normalisation doesn't make sense for single instances");
	}

	// This should probably be connected to trainData?
	public Instances determineOutputFormat(Instances inputFormat) {
		return new Instances(inputFormat, 0);
	}

	public void setTrainData(Instances data) { // Same as the constructor
		trainData = data;
		classIndex = data.classIndex();
		// Finds all the stats, doesnt cost much more really
		findStats(data);
	}

	public void setNormMethod(NormType n) {
		norm = n;
	}

	
	@Override
	public TimeSeriesInstances transform(TimeSeriesInstances inst) {
		
		double[][][] out = null;
		switch (norm) {
			case INTERVAL:
				out = intervalNorm(inst);
				break;
			case STD_NORMAL:
				out = standardNorm(inst);
				break;
		}

		return new TimeSeriesInstances(out, inst.getClassIndexes(), inst.getClassLabels());
	}

	/* Wont normalise the class value */
	public double[][][] intervalNorm(TimeSeriesInstances r) {
		double[][][] out = new double[r.numInstances()][][];
		int i =0;
		for (TimeSeriesInstance inst : r) {			
			out[i++] = ArrayUtilities.transposeMatrix(intervalNorm(inst));
		}

		return out;
	}

	public double[][] intervalNorm(TimeSeriesInstance r) {
		double[][] out = new double[r.getMaxLength()][];
		for (int j = 0; j < r.getMaxLength(); j++) {
			out[j] = TimeSeriesSummaryStatistics.intervalNorm(r.getVSliceArray(j), min[j], max[j]);
		}

		return out;
	}

	public double[][][] standardNorm(TimeSeriesInstances r) {
		double[][][] out = new double[r.numInstances()][][];

		int index=0;
		for(int i=0; i<r.numInstances(); i++){
			out[index] =  new double[r.getMaxLength()][];
			for (int j = 0; j < r.getMaxLength(); j++) {
				out[index][j] = TimeSeriesSummaryStatistics.standardNorm(r.get(i).getVSliceArray(j), mean[j], stdev[j]);
			}
			out[index++] = ArrayUtilities.transposeMatrix(out[index]);
		}
		return out;
	}

	public Instances transform(Instances inst) {
		Instances result = new Instances(inst);
		switch (norm) {
			case INTERVAL:
				intervalNorm(result);
				break;
			case STD_NORMAL:
				standardNorm(result);
				break;
		}
		return result;
	}

	/* Wont normalise the class value */
	public void intervalNorm(Instances r) {
		for (int i = 0; i < r.numInstances(); i++) {
			intervalNorm(r.instance(i));
		}
	}

	public void intervalNorm(Instance r) {
		for (int j = 0; j < r.numAttributes(); j++) {
			if (j != classIndex) {
				double x = r.value(j);
				r.setValue(j, (x - min[j]) / (max[j] - min[j]));
			}
		}
	}

	public void standardNorm(Instances r) {
		for (int j = 0; j < r.numAttributes(); j++) {
			if (j != classIndex) {
				for (int i = 0; i < r.numInstances(); i++) {
					double x = r.instance(i).value(j);
					r.instance(i).setValue(i, (x - mean[j]) / (stdev[j]));
				}
			}
		}
	}

	@Override
	public boolean isFit() {
		return isFit;
	}

	@Override
	public void fit(Instances data) {
		trainData = data;
		classIndex = data.classIndex();
		// Finds all the stats, doesnt cost much more really
		findStats(data);
		isFit = true;
	}

	@Override
	public void fit(TimeSeriesInstances data) {
		findStats(data);
		isFit = true;
	}


}
