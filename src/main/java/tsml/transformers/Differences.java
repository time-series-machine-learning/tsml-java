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

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

/* simple Filter that just creates a new series of differences order k.
 * The new series has k fewer attributes than the original
 * */
public class Differences implements Transformer {
	private int order = 1;
	String attName = "";

	public void setOrder(int m) {
		order = m;
	}

	private static final long serialVersionUID = 1L;

	public void setAttName(String s) {
		attName = s;
	}

	public Instances determineOutputFormat(Instances inputFormat) {
		// Set up instances size and format.
		int classIndexMod = (inputFormat.classIndex() >= 0 ? 1 : 0);
		ArrayList<Attribute> atts = new ArrayList<>();
		String name;
		for (int i = 0; i < inputFormat.numAttributes() - order - classIndexMod; i++) {
			name = attName + "Difference" + order + "_" + (i + 1);
			atts.add(new Attribute(name));
		}
		if (inputFormat.classIndex() >= 0) { // Classification set, set class
			// Get the class values as a fast vector
			Attribute target = inputFormat.attribute(inputFormat.classIndex());

			ArrayList<String> vals = new ArrayList<>();
			for (int i = 0; i < target.numValues(); i++)
				vals.add(target.value(i));
			atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
		}
		Instances result = new Instances("Difference" + order + inputFormat.relationName(), atts,
				inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}
		return result;
	}

	@Override
	public Instance transform(Instance inst) {
		// 1. Get series:
		double[] d = inst.toDoubleArray();
		// 2. Remove target class
		double[] temp;
		int c = inst.classIndex();
		if (c >= 0) {
			temp = new double[d.length - 1];
			System.arraycopy(d, 0, temp, 0, c);
			d = temp;
		}
		// 3. Create Difference series
		int classIndexMod = (c >= 0 ? 1 : 0);
		int numAtts = inst.numAttributes() - order - classIndexMod; //if have a classindex then make it one shorter.

		double[] diffs = calculateDifferences(d, numAtts);

		// Extract out the terms and set the attributes
		Instance newInst = new DenseInstance(diffs.length + classIndexMod);

		for (int j = 0; j < diffs.length; j++) {
			newInst.setValue(j, diffs[j]);
		}

		if (c >= 0)
			newInst.setValue(diffs.length, inst.classValue());

		return newInst;
	}

	private double[] calculateDifferences(double[] d, int numAtts) {
		double[] diffs = new double[numAtts];

		for (int j = 0; j < diffs.length; j++)
			diffs[j] = d[j] - d[j + order];
		return diffs;
	}

	public static void main(String[] args) {
		/**
		 * Debug code to test SummaryStats generation:
		 * 
		 * 
		 * try{ Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC
		 * Problems\\Beef\\Beef_TRAIN"); // Instances filter=new
		 * SummaryStats().process(test); SummaryStats m=new SummaryStats();
		 * m.setInputFormat(test); Instances filter=Filter.useFilter(test,m);
		 * System.out.println(filter); } catch(Exception e){
		 * System.out.println("Exception thrown ="+e); e.printStackTrace();
		 * 
		 * }
		 * 
		 */
	}

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for (TimeSeries ts : inst) {
            out[i++] = calculateDifferences(ts.toValueArray(), ts.getSeriesLength() - order);
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

}
