/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.transformers;

import java.io.FileReader;
import java.util.ArrayList;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/*
     * copyright: Anthony Bagnall
 * 
 * */
public class RunLength implements Transformer {
	private int maxRunLength = 50;
	private boolean useGlobalMean = true;
	private double globalMean = 5.5;

	public RunLength() {
	}

	public RunLength(int maxRL) {
		maxRunLength = maxRL;
	}

	public void setMaxRL(int m) {
		maxRunLength = m;
	}

	public void setGlobalMean(double d) {
		useGlobalMean = true;
		globalMean = d;
	}

	public void noGlobalMean() {
		useGlobalMean = false;
	}

	@Override
	public Instances determineOutputFormat(Instances inputFormat) {
		// Treating counts as reals
		ArrayList<Attribute> atts = new ArrayList<>();
		Attribute a;
		for (int i = 0; i < maxRunLength; i++) {
			a = new Attribute("RunLengthCount" + (i + 1));
			atts.add(a);
		}
		if (inputFormat.classIndex() >= 0) { // Classification set, set class
			// Get the class values as a fast vector
			Attribute target = inputFormat.attribute(inputFormat.classIndex());
			ArrayList<String> vals = new ArrayList<>(target.numValues());
			for (int i = 0; i < target.numValues(); i++)
				vals.add(target.value(i));
			atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));

		}
		Instances result = new Instances("RunLengths" + inputFormat.relationName(), atts, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0)
			result.setClassIndex(result.numAttributes() - 1);
		return result;
	}

	@Override
	public Instance transform(Instance inst) {
		// 1: Get series into an array, remove class value if present
		
		double[] d = InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst);
		double t = 0;
		if (useGlobalMean)
			t = globalMean;
		else { // Find average
			t = InstanceTools.mean(inst);
		}
		double[] histogram = create_data(d, t);

		Instance newInst = new DenseInstance(inst.numAttributes());

		// 3. Put run lengths and class value into instances
		for (int j = 0; j < histogram.length; j++)
			newInst.setValue(j, histogram[j]);

		if (inst.classIndex() >= 0)
			newInst.setValue(inst.numAttributes() - 1, inst.classValue());

		return newInst;
	}

	private double[] create_data(double[] d, double t) {
		// 2: Form histogram of run lengths: note missing values assumed in the same run
		double[] histogram = new double[d.length];
		int pos = 1;
		int length = 0;
		boolean u2 = false;
		boolean under = d[0] < t ? true : false;
		while (pos < d.length) {
			u2 = d[pos] < t ? true : false;
			// System.out.println("Pos ="+pos+" currentUNDER ="+under+" newUNDER = "+u2);
			if (Double.isNaN(d[pos]) || under == u2) {
				length++;
			} else {
				// System.out.println("Position "+pos+" has run length "+length);
				if (length < maxRunLength - 1)
					histogram[length]++;
				else
					histogram[maxRunLength - 1]++;
				under = u2;
				length = 0;
			}
			pos++;
		}
		if (length < maxRunLength - 1)
			histogram[length]++;
		else
			histogram[maxRunLength - 1]++;
		return histogram;
	}

	// Primitives version, assumes zero mean global, passes max run length
	public int[] processSingleSeries(double[] d, int mrl) {
		double mean = 0;
		int pos = 1;
		int length = 0;
		boolean u2 = false;
		int[] histogram = new int[mrl];
		boolean under = d[0] < mean ? true : false;
		while (pos < d.length) {
			u2 = d[pos] < mean ? true : false;
			if (under == u2) {
				length++;
			} else {
				if (length < mrl - 1)
					histogram[length]++;
				else
					histogram[mrl - 1]++;
				under = u2;
				length = 0;
			}
			pos++;
		}
		if (length < mrl - 1)
			histogram[length]++;
		else
			histogram[mrl - 1]++;

		return histogram;
	}

	// Test Harness
	public static void main(String[] args) {
		RunLength cp = new RunLength();
		cp.noGlobalMean();
		Instances data = null;
		String fileName = "C:\\Research\\Data\\Time Series Data\\Time Series Classification\\TestData\\TimeSeries_Train.arff";
		try {
			FileReader r;
			r = new FileReader(fileName);
			data = new Instances(r);

			data.setClassIndex(data.numAttributes() - 1);
			System.out.println(data);

			Instances newInst = cp.transform(data);
			System.out.println("\n" + newInst);
		} catch (Exception e) {
			System.out.println(" Error =" + e);
			StackTraceElement[] st = e.getStackTrace();
			for (int i = st.length - 1; i >= 0; i--)
				System.out.println(st[i]);

		}
	}

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i =0;
        for(TimeSeries ts : inst){
            out[i++] = create_data(ts.toValueArray(), useGlobalMean ? globalMean : TimeSeriesSummaryStatistics.mean(ts));
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }
	
	
}
