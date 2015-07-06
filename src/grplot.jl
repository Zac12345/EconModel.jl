function scatter(x, y;
                     bgcolor=0,
                     viewport=(0.1, 0.95, 0.1, 0.95),
                     scale=0,
                     grid=true,
                     linetype=GR.LINETYPE_SOLID,
                     markertype=GR.MARKERTYPE_DOT,
                     update=true)
           
           GR.clearws()
           
           
           if scale & GR.OPTION_X_LOG == 0
               xmin, xmax = GR.adjustrange(minimum(x), maximum(x))
           else
               xmin, xmax = (minimum(x), maximum(x))
           end
           if scale & GR.OPTION_Y_LOG == 0
               ymin, ymax = GR.adjustrange(minimum(y), maximum(y))
           else
               ymin, ymax = (minimum(y), maximum(y))
           end
           
           if scale & GR.OPTION_X_LOG == 0
               majorx = 5
               xtick = GR.tick(xmin, xmax) / majorx
           else
               xtick = majorx = 1
           end
           if scale & GR.OPTION_Y_LOG == 0
               majory = 5
               ytick = GR.tick(ymin, ymax) / majory
           else
               ytick = majory = 1
           end
           GR.setviewport(viewport[1], viewport[2], viewport[3], viewport[4])
           GR.setwindow(xmin, xmax, ymin, ymax)
           GR.setscale(scale)
           if bgcolor != 0
               GR.setfillintstyle(1)
               GR.setfillcolorind(bgcolor)
               GR.fillrect(xmin, xmax, ymin, ymax)
           end
           charheight = 0.024 * (viewport[4] - viewport[3])
           GR.setcharheight(charheight)
           if grid
                GR.grid(xtick, ytick, xmax, ymax, majorx, majory)
           end
           GR.axes(xtick, ytick, xmin, ymin, majorx, majory, 0.01)
           GR.axes(xtick, ytick, xmax, ymax, -majorx, -majory, -0.01)
           # GR.setlinetype(linetype)
           # GR.polyline(x, y)
           GR.polymarker(x, y)
           if markertype != GR.MARKERTYPE_DOT
               GR.setmarkertype(markertype)
               GR.polymarker(x, y)
           end
           if update
               GR.updatews()
           end

           if !GR.isinteractive()
               return GR.show()
           end
       end



function dplot(x::Array{Float64})
  kd = kde(x)
  EconModel.plot(kd.x,kd.density)
end
