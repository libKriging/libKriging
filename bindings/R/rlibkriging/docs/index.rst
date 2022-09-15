.. documentation master file, created by sphinx-quickstart 
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

reStructuredText
================================

.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

This main document is in `'reStructuredText' ("rst") format
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_,
which differs in many ways from standard markdown commonly used in R packages.
``rst`` is richer and more powerful than markdown. The remainder of this main
document demonstrates some of the features, with links to additional ``rst``
documentation to help you get started. The definitive argument for the benefits
of ``rst`` over markdown is the `official language format documentation
<https://www.python.org/dev/peps/pep-0287/>`_, which starts with a very clear
explanation of the `benefits
<https://www.python.org/dev/peps/pep-0287/#benefits>`_.

Examples
--------

All of the following are defined within the ``docs/index.rst`` file. Here is
some :red:`coloured` text which demonstrates how raw HTML commands can be
incorporated. The following are examples of ``rst`` "admonitions":

.. note::

    Here is a note

    .. warning::

        With a warning inside the note

.. seealso::

    The full list of `'restructuredtext' directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_ or a similar list of `admonitions <https://restructuredtext.documatt.com/admonitions.html>`_.

.. centered:: This is a line of :red:`centered text`

.. hlist::
   :columns: 3

   * and here is
   * A list of
   * short items
   * that are
   * displayed
   * in 3 columns

The remainder of this document shows three tables of contents for the main
``README`` (under "Introduction"), and the vignettes and R directories of
a package. These can be restructured any way you like by changing the main
``docs/index.rst`` file. The contents of this file -- and indeed the contents
of any `readthedocs <https://readthedocs.org>`_ file -- can be viewed by
clicking *View page source* at the top left of any page.

.. toctree::
   :maxdepth: 1
   :caption: Introduction:
