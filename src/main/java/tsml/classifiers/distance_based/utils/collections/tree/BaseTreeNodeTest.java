package tsml.classifiers.distance_based.utils.collections.tree;

import org.junit.Assert;
import org.junit.Test;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class BaseTreeNodeTest {

    @Test
    public void testHeightAndSize() {
        //  a
        //  |
        //  b
        // / \
        // c d
        // |
        // e
        TreeNode<String> a = new BaseTreeNode<>("a");
        TreeNode<String> b = new BaseTreeNode<>("b");
        TreeNode<String> c = new BaseTreeNode<>("c");
        TreeNode<String> d = new BaseTreeNode<>("d");
        TreeNode<String> e = new BaseTreeNode<>("e");
        a.addChild(b);
        b.addChild(c);
        b.addChild(d);
        c.addChild(e);
        //            e.addChild(e); // todo check this errs
        Assert.assertEquals(a.size(), 5);
        Assert.assertEquals(a.height(), 4);
    }

    //        @Test(expected = IllegalArgumentException.class)
    //        public void testAddSelfAsChild() {
    //            BaseTreeNode<String> node = new BaseTreeNode<>("a");
    //            node.addChild(node);
    //        }
}
