/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.collections.cache;

public class SymmetricBiCache<A, B> extends BiCache<A, A, B> {

    // todo cache state read / write

    @Override
    public B get(final A firstKey, final A secondKey) {
        B result = super.get(firstKey, secondKey);
        if(result == null) {
            result = super.get(secondKey, firstKey);
        }
        return result;
    }

    @Override
    public void put(final A firstKey, final A secondKey, final B value) {
        super.put(firstKey, secondKey, value);
    }

    @Override
    public boolean remove(final A firstKey, final A secondKey) {
        boolean removed = super.remove(firstKey, secondKey);
        if(!removed) {
            removed = super.remove(secondKey, firstKey);
        }
        return removed;
    }
}
