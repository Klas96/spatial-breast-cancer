-- Lua filter: map custom STAR Protocols LaTeX commands to Word-friendly output

local function strong(s)  return pandoc.Strong(pandoc.Str(s)) end
local function space()    return pandoc.Space() end

-- Inline handler: \critical, \notepar, \pausept
function RawInline(el)
  if el.format ~= "latex" then return nil end
  local t = el.text

  local label, body
  for pat, lbl in pairs({
    ["\\critical%{(.-)%}"]  = "CRITICAL:",
    ["\\notepar%{(.-)%}"]   = "NOTE:",
    ["\\pausept%{(.-)%}"]   = "PAUSE POINT:",
  }) do
    body = t:match(pat)
    if body then label = lbl; break end
  end

  if label then
    return { strong(label), space(),
             pandoc.RawInline("markdown", body) }
  end
end

-- Block handler: tcolorbox timing blocks and highlights box
function RawBlock(el)
  if el.format ~= "latex" then return nil end
  local t = el.text

  -- \timing{...}
  local inner = t:match("\\timing%{(.-)%}$")
  if not inner then
    inner = t:match("\\timing%{(.+)%}")
  end
  if inner then
    return pandoc.BlockQuote({
      pandoc.Para({strong("Timing:"), space(),
                   pandoc.RawInline("markdown", inner)})
    })
  end
end

-- Para-level: strip remaining unknown raw latex so content shows through
function Para(el)
  local new = {}
  for _, inline in ipairs(el.content) do
    if inline.t == "RawInline" and inline.format == "latex" then
      -- keep the argument of known single-arg commands
      local body = inline.text:match("^\\%a+%{(.-)%}$")
      if body then
        table.insert(new, pandoc.RawInline("markdown", body))
      end
      -- else drop silently
    else
      table.insert(new, inline)
    end
  end
  el.content = new
  return el
end
