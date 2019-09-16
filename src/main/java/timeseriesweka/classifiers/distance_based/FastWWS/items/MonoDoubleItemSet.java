/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 * 
 * This file is part of FastWWSearch.
 * 
 * FastWWSearch is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * FastWWSearch is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FastWWSearch.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package timeseriesweka.classifiers.distance_based.FastWWS.items;
import static java.lang.Math.abs;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class MonoDoubleItemSet extends Itemset implements java.io.Serializable {
	private static final long serialVersionUID = 5103879297281957601L;
	
	public double value;
	
	public MonoDoubleItemSet(double value){
		this.value = value;
	}
	
	@Override
	public Itemset clone() {
		return new MonoDoubleItemSet(value);
	}

	@Override
	public double distance(Itemset o) {
		MonoDoubleItemSet o1 = (MonoDoubleItemSet)o;
		return abs(o1.value-value);
	}

	@Override
	public Itemset mean(Itemset[] tab) {
		if (tab.length < 1) {
			throw new RuntimeException("Empty tab");
		}
		double sum = 0.0;
		for (Itemset itemset : tab) {
			MonoDoubleItemSet item = (MonoDoubleItemSet)itemset;
			sum += item.value;
		}
		return new MonoDoubleItemSet(sum / tab.length);
	}

	@Override
	public String toString() {
		return new Double(value).toString();
	}
	
	public double getValue(){
		return value;
	}
}
