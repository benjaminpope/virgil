# `virgil.legacy`

Legacy OIFITS tools derived from [ImPlaneIA](https://github.com/anand0xff/ImPlaneIA),
working on its dictionary layout (phases in **degrees**). They are not imported
by `import virgil`. New code should use [`virgil.oifits`](oifits.md) and
[`OIData`](oidata.md).

::: virgil.legacy.oifits_implaneia
    options:
      members:
        - save
        - load
        - GetWavelength
        - Format_STAINDEX_V2
        - Format_STAINDEX_T3
        - rad2mas
