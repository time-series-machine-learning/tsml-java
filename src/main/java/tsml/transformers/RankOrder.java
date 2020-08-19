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

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities.Capability;
import weka.filters.*;

import java.util.*;

import org.apache.commons.lang3.NotImplementedException;

import tsml.data_containers.TSCapabilities;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.DenseInstance;

/*
     * copyright: Anthony Bagnall
 * */
public class RankOrder implements Transformer {

	public double[][] ranks;
	public int numAtts = 0;
	private boolean normalise = true;

	public void setNormalise(boolean f) {
		normalise = f;
	};

	@Override
	public Instance transform(Instance inst) {
		throw new NotImplementedException(
				"Single instance transformation does not make sense for rank order, which is column wise ranking!");
	}

	@Override
	public Instances transform(Instances inst) {
		// Set input instance format

		Instances result = new Instances(determineOutputFormat(inst), 0);
		rankOrder(inst);
		// Stuff into new set of instances
		for (int i = 0; i < inst.numInstances(); i++) {
			// Create a deep copy, think this is necessary to maintain meta data?
			Instance in = new DenseInstance(inst.instance(i));
			// Reset to the ranks
			for (int j = 0; j < numAtts; j++)
				in.setValue(j, ranks[i][j]);
			result.add(in);
		}
		return result;
	}

	protected class Pair implements Comparable {
		int pos;
		double val;

		public Pair(int p, double d) {
			pos = p;
			val = d;
		}

		public int compareTo(Object c) {
			if (val > ((Pair) c).val)
				return 1;
			if (val < ((Pair) c).val)
				return -1;
			return 0;
		}

	}

	public void rankOrder(Instances inst) {
		numAtts = inst.numAttributes();
		int c = inst.classIndex();
		if (c > 0)
			numAtts--;
		// If a classification problem it is assumed the class attribute is the last one
		// in the instances
		Pair[][] d = new Pair[numAtts][inst.numInstances()];
		for (int j = 0; j < inst.numInstances(); j++) {
			Instance x = inst.instance(j);
			for (int i = 0; i < numAtts; i++)
				d[i][j] = new Pair(j, x.value(i));
		}
		// Form rank order data set (in transpose of sorted array)
		// Sort each array of Pair
		for (int i = 0; i < numAtts; i++)
			Arrays.sort(d[i]);
		ranks = new double[inst.numInstances()][numAtts];
		for (int j = 0; j < inst.numInstances(); j++)
			for (int i = 0; i < numAtts; i++)
				ranks[d[i][j].pos][i] = j;

	}

	@Override
	public Instances determineOutputFormat(Instances inputFormat) {
		Instances result = new Instances(inputFormat, 0);
		return result;
	}

	public TSCapabilities getTSCapabilities() {
		TSCapabilities result = Transformer.super.getTSCapabilities();
		/*result.enableAllAttributes();
		result.enableAllClasses();
		result.enable(Capability.NO_CLASS); // filter doesn't need class to be set*/
		return result;
	}

	public static void main(String[] args) {

	}

	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
		// TODO Auto-generated method stub
		return null;
	}



}
