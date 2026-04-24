
import folium
import branca.colormap as cm

def folium_colored_points(
    df,
    value_col="coefficient",
    lat_col="lat",
    lon_col="lon",
    zoom_start=6
):
    # Center map
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Color scale
    colormap = cm.LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=df[value_col].min(),
        vmax=df[value_col].max(),
        caption=value_col
    )

    # Add points
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            color=colormap(row[value_col]),
            fill=True,
            fill_color=colormap(row[value_col]),
            fill_opacity=0.8,
            popup=f"{value_col}: {row[value_col]}"
        ).add_to(m)

    colormap.add_to(m)
    return m

