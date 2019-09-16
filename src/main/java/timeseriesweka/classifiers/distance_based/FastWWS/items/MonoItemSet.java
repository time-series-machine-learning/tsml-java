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

import java.util.HashMap;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class MonoItemSet extends Itemset {
	public String letter;
	
	public MonoItemSet(String letter){
		this.letter = letter;
	}
	@Override
	public Itemset clone() {
		return new MonoItemSet(new String(letter));
	}

	@Override
	public double distance(Itemset o) {
		MonoItemSet mono = (MonoItemSet)o;
		return (letter.equals(mono.letter))?0.0:1.0;
	}

	@Override
	/**
	 * vote
	 */
	public Itemset mean(Itemset[] tab) {
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		int maxCount = 0;
		String maxKey=null;
		for(Itemset itemset:tab){
			String key = ((MonoItemSet)itemset).letter;
			Integer count = map.get(key);
			Integer newCount ;
			if(count == null){
				newCount = 1;
			}else{
				newCount = count+1;
			}
			if(newCount>maxCount){
				maxCount = newCount;
				maxKey = key;
			}
			map.put(key, newCount);
		}
		return new MonoItemSet(maxKey);
	}

	@Override
	public String toString() {
		return letter;
	}
}
