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
 
package tsml.classifiers.distance_based.utils.collections.tree;

import java.util.AbstractList;
import java.util.Collection;
import java.util.Iterator;

/**
 * A general tree data structure. No context of nodes at all, just a simple hierarchy.
 *
 * Contributors: goastler
 */

public class BaseTree<A> extends AbstractList<A> implements Tree<A> {

    public BaseTree() {
        clear();
    }

    private TreeNode<A> root;

    @Override
    public TreeNode<A> getRoot() {
        return root;
    }

    @Override
    public void setRoot(TreeNode<A> root) {
        this.root = root;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{root=" + root + "}";
    }

    public A get(int i) {
        throw new UnsupportedOperationException("get not implemented for tree");
    }

    @Override public int size() {
        return Tree.super.size();
    }

    @Override public boolean equals(final Object o) {
        return this == o;
    }
}
