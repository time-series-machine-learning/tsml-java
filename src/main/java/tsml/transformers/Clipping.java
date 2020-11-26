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
import java.util.Arrays;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.TimeSeriesStatsTools;
import utilities.InstanceTools;
import weka.core.*;

public class Clipping implements Transformer {
	boolean useMean = true;
	boolean useRealAttributes = true;

	public void setUseRealAttributes(boolean f) {
		useRealAttributes = f;
	}

	@Override
	public Instances determineOutputFormat(Instances inputFormat) {
		// Must convert all attributes to binary.
		Attribute a;
		FastVector<String> fv = new FastVector<>();
		if (!useRealAttributes) {
			fv.addElement("0");
			fv.addElement("1");
		}
		FastVector<Attribute> atts = new FastVector<>();

		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			// System.out.println(" Create Attribute "+i);
			if (i != inputFormat.classIndex()) {
				if (!useRealAttributes)
					a = new Attribute("Clipped" + inputFormat.attribute(i).name(), fv);
				else
					a = new Attribute("Clipped" + inputFormat.attribute(i).name());
			} else
				a = inputFormat.attribute(i);
			atts.addElement(a);
			// System.out.println(" Add Attribute "+i);
			// result.insertAttributeAt(a,i);
		}
		Instances result = new Instances("Clipped" + inputFormat.relationName(), atts, inputFormat.numInstances());
		// System.out.println(" Output format ="+result);
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}
		return result;
	}

	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
		//could do this across all dimensions.
		double[][] out = new double[inst.getNumDimensions()][];
		int i = 0;
		for(TimeSeries ts : inst){
			double mean = TimeSeriesStatsTools.mean(ts);
			out[i++] = ts.streamValues().map(e -> e < mean ? 0.0 : 1.0).toArray();
		}
		
		//create a new output instance with the ACF data.
		return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
	}

	@Override
	public Instance transform(Instance inst) {
		Instance newInst;


		double average = InstanceTools.mean(inst);

		if (!useRealAttributes) {
			newInst = new DenseInstance(inst.numAttributes());
			for (int j = 0; j < inst.numAttributes(); j++) {
				if (inst.isMissing(j))
					newInst.setValue(j, "?");

				if (j != inst.classIndex()) 
					newInst.setValue(j, inst.value(j) < average ? "0" : "1");
				else
					newInst.setValue(j, inst.stringValue(j));
			}
			return newInst;
		} else {
			newInst = new DenseInstance(inst.numAttributes());
			for (int j = 0; j < inst.numAttributes(); j++) {
				if (inst.isMissing(j)) continue; //skip/set to 0.0 if it's a missing value.

				if (j != inst.classIndex()) 
					newInst.setValue(j, inst.value(j) < average ? 0 : 1);
				else
					newInst.setValue(j, inst.value(j));
				
			}
		}
		return newInst;
	}

	public static void main(String[] args) {
		Clipping cp = new Clipping();
		Instances data = null;
		String fileName = "C:\\Research\\Data\\Time Series Data\\Time Series Classification\\TestData\\TimeSeries_Train.arff";
		try {
			FileReader r;
			r = new FileReader(fileName);
			data = new Instances(r);

			data.setClassIndex(data.numAttributes() - 1);

			System.out.println(" Class type numeric =" + data.attribute(data.numAttributes() - 1).isNumeric());
			System.out.println(" Class type nominal =" + data.attribute(data.numAttributes() - 1).isNominal());

			Instances newInst = cp.transform(data);
			System.out.println(newInst);
		} catch (Exception e) {
			System.out.println(" Error =" + e);
			StackTraceElement[] st = e.getStackTrace();
			for (int i = st.length - 1; i >= 0; i--)
				System.out.println(st[i]);

		}
	}

}
