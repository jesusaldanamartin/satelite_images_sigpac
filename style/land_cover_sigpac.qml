<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" minScale="1e+08" version="3.22.4-Białowieża" hasScaleBasedVisibilityFlag="0" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal mode="0" fetchMode="0" enabled="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option name="WMSBackgroundLayer" value="false" type="bool"/>
      <Option name="WMSPublishDataSourceUrl" value="false" type="bool"/>
      <Option name="embeddedWidgets/count" value="0" type="int"/>
      <Option name="identify/format" value="Value" type="QString"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option name="name" value="" type="QString"/>
      <Option name="properties"/>
      <Option name="type" value="collection" type="QString"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" enabled="false"/>
    </provider>
    <rasterrenderer alphaBand="-1" band="1" nodataColor="" opacity="1" type="paletted">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry value="1" color="#0032c8" alpha="255" label="Corrientes y superficies de agua"/>
        <paletteEntry value="2" color="#fa0000" alpha="255" label="Viales"/>
        <paletteEntry value="3" color="#f096ff" alpha="255" label="Citricos-Frutal"/>
        <paletteEntry value="4" color="#f096ff" alpha="255" label="Citricos"/>
        <paletteEntry value="5" color="#f096ff" alpha="255" label="Citricos-Frutal de cascara"/>
        <paletteEntry value="6" color="#f096ff" alpha="255" label="Citricos-Viñedo"/>
        <paletteEntry value="7" color="#fa0000" alpha="255" label="Edificaciones"/>
        <paletteEntry value="8" color="#fa0000" alpha="255" label="Elemento del Paisaje"/>
        <paletteEntry value="9" color="#f096ff" alpha="255" label="Frutal de Cascara-Frutal"/>
        <paletteEntry value="10" color="#f096ff" alpha="255" label="Frutal de Cascara-Olivar"/>
        <paletteEntry value="11" color="#648c00" alpha="255" label="Forestal"/>
        <paletteEntry value="12" color="#f096ff" alpha="255" label="Frutal de Cascara"/>
        <paletteEntry value="13" color="#f096ff" alpha="255" label="Frutal de Cascara-Viñedo"/>
        <paletteEntry value="14" color="#f096ff" alpha="255" label="Frutal"/>
        <paletteEntry value="15" color="#b4b4b4" alpha="255" label="Improductivo"/>
        <paletteEntry value="16" color="#0096a0" alpha="255" label="Imvernadero y cultivos bajo plastico"/>
        <paletteEntry value="17" color="#f096ff" alpha="255" label="Olivar-Citricos"/>
        <paletteEntry value="18" color="#f096ff" alpha="255" label="Olivar-Frutal"/>
        <paletteEntry value="19" color="#f096ff" alpha="255" label="Olivar"/>
        <paletteEntry value="20" color="#ffbb22" alpha="255" label="Pasto Arbolado"/>
        <paletteEntry value="21" color="#ffbb22" alpha="255" label="Pasto Arbustivo"/>
        <paletteEntry value="22" color="#ffff4c" alpha="255" label="Pastizal"/>
        <paletteEntry value="23" color="#f096ff" alpha="255" label="Tierra Arable"/>
        <paletteEntry value="24" color="#f096ff" alpha="255" label="Huerta"/>
        <paletteEntry value="25" color="#f096ff" alpha="255" label="Frutal-Viñedo"/>
        <paletteEntry value="26" color="#f096ff" alpha="255" label="Viñedo"/>
        <paletteEntry value="27" color="#f096ff" alpha="255" label="Olivar-Viñedo"/>
        <paletteEntry value="28" color="#fa0000" alpha="255" label="Zona Concentrada"/>
        <paletteEntry value="29" color="#fa0000" alpha="255" label="Zona Urbana"/>
        <paletteEntry value="30" color="#fa0000" alpha="255" label="Zona Censurada"/>
      </colorPalette>
      <colorramp name="[source]" type="randomcolors">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" invertColors="0" colorizeStrength="100" saturation="0" colorizeOn="0" colorizeBlue="128" grayscaleMode="0" colorizeRed="255"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
