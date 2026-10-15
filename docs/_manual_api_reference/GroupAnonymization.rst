.. type:: GroupAnonymization
    :no-contents-entry:
    :annotation: : (Collection[Collection], Algorithm) -> Collection[Collection]

    Prototype of a group anonymization method for local recoding algorithms.

    Local recoding algorithms typically split the original records into groups.
    Then, a group anonymization method (implementation of this type) is applied 
    to make records become indistinguishable in their respective group.

    :param group: Group of records to be anonymized.
    :type group: Collection[Collection]
    :param algorithm: Algorithm instance containing necessary properties that can be used for anonymization. Typically, this is the algorithm providing `group`.
    :type algorithm: Algorithm

    :returns: *Collection[Collection]* -- The anonymized group.

    .. seealso::

        :class:`LocalRecodingAlgorithm`
            Abstract class for local recoding-based k-anonymization algorithms.

        :class:`GroupAnonymizationBuiltIn`
            A set of built-in ``GroupAnonymization``.