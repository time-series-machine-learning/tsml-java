package tsml.classifiers.distance_based.utils.collections.tree;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class BaseTreeNodeTest {
    
    private TreeNode<String> empty;
    private TreeNode<String> root;
    private TreeNode<String> left;
    private TreeNode<String> leftLeft;
    private TreeNode<String> leftMiddle;
    private TreeNode<String> leftRight;
    private TreeNode<String> middle;
    private TreeNode<String> middleLeft;
    private TreeNode<String> middleMiddle;
    private TreeNode<String> middleRight;
    private TreeNode<String> right;
    private TreeNode<String> rightLeft;
    private TreeNode<String> rightMiddle;
    private TreeNode<String> rightRight;
    
    @Before
    public void before() {
        root = new BaseTreeNode<>("root");
        left = new BaseTreeNode<>("left", root);
        leftLeft = new BaseTreeNode<>("left", left);
        leftMiddle = new BaseTreeNode<>("middle", left);
        leftRight = new BaseTreeNode<>("right", left);
        middle = new BaseTreeNode<>("middle", root);
        middleLeft = new BaseTreeNode<>("middleLeft", middle);
        middleMiddle = new BaseTreeNode<>("middleMiddle", middle);
        middleRight = new BaseTreeNode<>("middleRight", middle);
        right = new BaseTreeNode<>("right", root);
        rightLeft = new BaseTreeNode<>("rightLeft", right);
        rightMiddle = new BaseTreeNode<>("rightMiddle", right);
        rightRight = new BaseTreeNode<>("rightRight", right);
        empty = new BaseTreeNode<>("empty");
    }
    
    @Test
    public void testHeight() {
        Assert.assertEquals(3, root.height());
        Assert.assertEquals(2, left.height());
        Assert.assertEquals(1, leftLeft.height());
        Assert.assertEquals(1, empty.height());
    }

    @Test
    public void testNodeCount() {
        Assert.assertEquals(13, root.numNodes());
        Assert.assertEquals(4, left.numNodes());
        Assert.assertEquals(1, leftLeft.numNodes());
        Assert.assertEquals(1, empty.numNodes());
    }

    @Test
    public void testSize() {
        Assert.assertEquals(3, root.size());
        Assert.assertEquals(3, left.size());
        Assert.assertEquals(0, leftLeft.size());
        Assert.assertEquals(0, empty.size());
    }

    @Test
    public void testGetAndSetValue() {
        String value = root.getValue();
        Assert.assertEquals("root", value);
        String newValue = "something";
        root.setValue(newValue);
        Assert.assertEquals(newValue, root.getValue());
    }
    
    @Test
    public void testSetParent() {
        Assert.assertNull(empty.getParent());
        empty.setParent(root);
        Assert.assertEquals(root, empty.getParent());
        Assert.assertEquals(empty, root.get(3));
        Assert.assertEquals(14, root.numNodes());
    }

    @Test
    public void testAddChild() {
        Assert.assertNull(empty.getParent());
        root.add(empty);
        Assert.assertEquals(root, empty.getParent());
        Assert.assertEquals(empty, root.get(3));
        Assert.assertEquals(14, root.numNodes());
    }
    
    @Test
    public void testRemoveChild() {
        Assert.assertEquals(newArrayList(left, middle, right), root);
        root.remove(1);
        Assert.assertEquals(newArrayList(left, right), root);
        Assert.assertNull(middle.getParent());
        Assert.assertEquals(9, root.numNodes());
    }
    
    @Test
    public void testEmpty() {
        Assert.assertTrue(empty.isLeaf());
        Assert.assertEquals(1, empty.numNodes());
        Assert.assertEquals(1, empty.height());
        Assert.assertEquals(0, empty.size());
    }

    @Test
    public void testClear() {
        Assert.assertEquals(newArrayList(left, middle, right), root);
        root.clear();
        Assert.assertEquals(0, root.size());
        Assert.assertNull(middle.getParent());
        Assert.assertNull(left.getParent());
        Assert.assertNull(right.getParent());
        Assert.assertEquals(1, root.numNodes());
        Assert.assertEquals(0, root.size());
        Assert.assertEquals(1, root.height());
    }

    @Test
    public void testSet() {
        Assert.assertEquals(newArrayList(left, middle, right), root);
        root.set(0, empty);
        Assert.assertEquals(newArrayList(empty, middle, right), root);
        Assert.assertEquals(10, root.numNodes());
        Assert.assertEquals(3, root.size());
        Assert.assertEquals(3, root.height());
    }


}
