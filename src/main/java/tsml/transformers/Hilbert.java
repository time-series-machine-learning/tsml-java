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

import java.io.File;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.*;

/*
     * copyright: Anthony Bagnall
	 * @author Aaron Bostrom
 * */
public class Hilbert implements Transformer {

	public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
		FastVector<Attribute> atts = new FastVector<>();

		for (int i = 0; i < inputFormat.numAttributes() - 1; i++) {
			// Add to attribute list
			String name = "Hilbert" + i;
			atts.addElement(new Attribute(name));
		}
		// Get the class values as a fast vector
		Attribute target = inputFormat.attribute(inputFormat.classIndex());

		FastVector<String> vals = new FastVector<>(target.numValues());
		for (int i = 0; i < target.numValues(); i++)
			vals.addElement(target.value(i));
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
		Instances result = new Instances("Hilbert" + inputFormat.relationName(), atts, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}

		return result;
	}

	@Override
	public Instances transform(Instances data) {
		// for k=1 to n: f_k = sum_{i=1}^n f_i cos[(k-1)*(\pi/n)*(i-1/2)]
		// Assumes the class attribute is in the last one for simplicity
		Instances result = determineOutputFormat(data);
		for (Instance inst : data)
			result.add(transform(inst));

		return result;
	}

	@Override
	public Instance transform(Instance inst) {
		int n = inst.numAttributes() - 1;
		Instance newInst = new DenseInstance(inst.numAttributes());
		for (int k = 0; k < n; k++) {
			double fk = 0;
			for (int i = 0; i < n; i++) {
				if (i != k)
					fk += inst.value(i) / (k - i);
			}
			newInst.setValue(k, fk);
		}
		newInst.setValue(inst.classIndex(), inst.classValue());
		return newInst;
	}

	public static void main(String[] args) {
		String local_path = "D:\\Work\\Data\\Univariate_ts\\"; // Aarons local path for testing.
		String dataset_name = "ChinaTown";
		Instances train = DatasetLoading
				.loadData(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.ts");
		Instances test = DatasetLoading
				.loadData(local_path + dataset_name + File.separator + dataset_name + "_TEST.ts");
		Hilbert hTransform = new Hilbert();
		Instances out_train = hTransform.transform(train);
		Instances out_test = hTransform.transform(test);
		System.out.println(out_train.toString());
		System.out.println(out_test.toString());
	}

	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
		double[][] out = new double[inst.getNumDimensions()][];
		int index = 0;
		for (TimeSeries ts : inst) {
			int n = ts.getSeriesLength();
			out[index] = new double[n];
			for (int k = 0; k < n; k++) {
				double fk = 0;
				for (int i = 0; i < n; i++) {
					if (i != k)
						fk += ts.getValue(i) / (k - i);
				}
				out[index][k] = fk;
			}
		}
		return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
	}

}
