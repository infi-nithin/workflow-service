local cjson = require "cjson.safe"

local cookie = ngx.var.http_cookie
local token = cookie and cookie:match("access_token=([^;]+)")

if not token then
    ngx.status = 401
    ngx.say('{"error": "No token"}')
    return ngx.exit(401)
end

local parts = {}
for part in string.gmatch(token, "([^.]+)") do table.insert(parts, part) end

if #parts >= 2 then
    local payload_raw = parts[2]
    payload_raw = payload_raw:gsub("-", "+"):gsub("_", "/")
    local padding = #payload_raw % 4
    if padding > 0 then payload_raw = payload_raw .. string.rep("=", 4 - padding) end
    
    local decoded = ngx.decode_base64(payload_raw)
    local data = cjson.decode(decoded)

if data then
    local roles = ""
    if data.resource_access and data.resource_access["public-client"] then
        roles = table.concat(data.resource_access["public-client"].roles or {}, ",")
    end
    local mf_scope = ""
    if data.scope then
        for s in string.gmatch(data.scope, "%S+") do
            if s == "MutualFunds" then
                mf_scope = "MutualFunds"
                break
            end
        end
    end

    ngx.req.set_header("Authorization", "Bearer " .. token)
    ngx.req.set_header("X-User-Id", data.sub or "")
    ngx.req.set_header("X-Username", data.preferred_username or "")
    ngx.req.set_header("X-User-Roles", roles)
    ngx.req.set_header("X-User-Scope", mf_scope)
end

end